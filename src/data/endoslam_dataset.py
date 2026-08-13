"""
EndoSLAM stomach-subset dataset loader.

Real layout, confirmed against the Kaggle mirror (mcocoz/endoslam) via
os.walk on 2026-08-13 -- very different from the originally-assumed
{organ}/{camera}/{sequence}/frames pattern:

    {root}/Cameras/{HighCam,LowCam}/Stomach-{I,II,III}/TumorfreeTrajectory_{1-4}/Frames/*.jpg
    {root}/Cameras/{HighCam,LowCam}/Stomach-{I,II,III}/TumorfreeTrajectory_{1-4}/Poses/*.xlsx
    {root}/UnityCam/Stomach/Frames/*
    {root}/UnityCam/Stomach/Pixelwise Depths/*
    {root}/UnityCam/Stomach/Poses/*

Real cameras (HighCam/LowCam) nest under Cameras/ with per-specimen organ
folders (Stomach-I/II/III) and 4 trajectories each; the synthetic UnityCam
sits at the top level as a single flat sequence (no specimen/trajectory
split). Also note: on newer Kaggle kernels the dataset mount itself can be
nested under an extra layer (e.g. /kaggle/input/datasets/<owner>/<slug>)
instead of the classic /kaggle/input/<slug> -- see find_endoslam_root() in
the validation notebook.

Pose format, confirmed 2026-08-13 via the phase3a_pose_explore Kaggle kernel
(see PROGRESS.md "Pose format -- confirmed facts" for the full findings) --
nobody had opened a real pose file before that kernel, both formats below
were guesses until then:

- Real-camera poses: one .xlsx per trajectory (e.g.
  "low_high_pose_stom2_teste2_low_images.xlsx"), single sheet, columns
  ImageFrame/Pose_Index/trans_x/trans_y/trans_z/quot_x/quot_y/quot_z/quot_w
  (quaternion + translation, meters). Row count does NOT reliably equal
  frame count, and ImageFrame is NOT a 0-based row position -- it's the
  original video frame index, and matches the zero-padded number in each
  frame's filename (frame_NNNNNN.jpg) exactly. Alignment is therefore by
  parsing that number and joining on ImageFrame, not positional order.
- UnityCam poses: NOT .xlsx -- a single
  UnityCam/Stomach/Poses/stomach_position_rotation.csv for the whole
  sequence, columns tX/tY/tZ/rX/rY/rZ/rW/time(s) (quaternion + translation,
  but in Unity world units -- a different scale than real-camera poses, do
  not assume the same coordinate convention). No frame-index/name column
  exists, so alignment here is positional (row i <-> frame i), truncated to
  min(frame_count, pose_row_count) since the two counts don't quite match
  (1544 rows vs 1548 frames in the sampled sequence). The real file's very
  last row is also a partial write (NaNs in rY/rZ/rW/time(s)) -- dropped by
  _load_unitycam_poses() since it's confined to the tail and so can't shift
  positional alignment for any earlier row.

Internal representation, once loaded: a 4x4 SE(3) matrix (float32) per
frame, regardless of source format -- matches eval.pose_metrics (ATE/RPE
are conventionally computed on SE(3)) and keeps rotation-representation
differences between the two camera types out of downstream code.

Depth: per-pixel depth maps exist only under UnityCam/Stomach/Pixelwise Depths/
-- HighCam/LowCam have no depth GT, matching the paper. Depth-to-frame
pairing is by sorted position (index i in Frames <-> index i in Pixelwise
Depths), not matching filenames -- the two dirs don't share a naming scheme.

Depth format, confirmed 2026-08-13 via the phase3b_depth_explore Kaggle
kernel: files are 8-bit 4-channel (RGBA) PNGs (e.g. "aov_image_0000.png",
"AOV" = Arbitrary Output Variable, a render-engine term), all four channels
identical -- __getitem__ takes channel 0 only. Values observed in [1, 72]
(not spanning the full 0-255 range) -- absolute units/scale are NOT
confirmed (no official EndoSLAM documentation found describing the
Unity depth-export encoding; a "raw millimeters" hypothesis is physically
plausible for stomach endoscopy working distances and is consistent with
config.yaml's fusion.depth_trunc: 0.15m, but is unverified). This is fine
for Phase 3 training: the official EndoSLAM repo's own eval_depth.py
applies median-ratio scaling between predicted and GT depth before
computing AbsRel/RMSE/delta1, meaning the reference methodology already
treats predicted-vs-GT depth as scale-ambiguous rather than assuming a
known absolute unit -- so Phase 3 trains directly against the raw pixel
values with a plain masked loss, and true-metric calibration (if ever
needed) is a Phase 4 fusion-time concern, not a Phase 3 training-time one.
"""

import os
import re
import glob
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

_FRAME_NUMBER_RE = re.compile(r"(\d+)")


def _pose_matrix(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """translation: (3,), quat_xyzw: (4,) in [x, y, z, w] order -> (4, 4) SE(3) matrix."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = translation
    return mat


@dataclass
class FrameSample:
    image_path: str
    pose: np.ndarray | None      # (4, 4) SE(3) matrix; None only if the frame had no matching pose row
    depth_path: str | None       # None for HighCam/LowCam
    camera: str                  # "UnityCam" | "HighCam" | "LowCam"
    sequence_id: str
    frame_idx: int


class EndoSLAMStomachDataset(Dataset):
    """
    Returns sequences of consecutive frames (length = context_window) rather
    than single frames, since the reconstruction model needs temporal
    context. DarkIR-lite training can just flatten these back to single
    frames -- see `flatten_for_enhancement()` below.
    """

    def __init__(self, config: dict, split: str, cameras: list[str], context_window: int = 8):
        self.root = config["data"]["root"]
        self.organ = config["data"]["organ"]
        self.image_size = tuple(config["data"]["image_size"])
        self.context_window = context_window
        self.cameras = cameras
        self.split = split
        self.synthetic_cam = config["data"]["synthetic_cam"]

        self.sequences = self._index_sequences()
        self._apply_split(config["data"]["train_split"], config["data"]["val_split"], config["data"]["seed"])
        self.windows = self._build_windows()

    def _index_sequences(self) -> dict:
        """See the module docstring for the confirmed real folder layout."""
        organ_name = self.organ.capitalize()  # config has "stomach" -> real folders are "Stomach*"
        sequences = {}
        for cam in self.cameras:
            if cam == "UnityCam":
                self._index_unitycam(organ_name, sequences)
            else:
                self._index_real_camera(cam, organ_name, sequences)
        return sequences

    def _index_real_camera(self, cam: str, organ_name: str, sequences: dict) -> None:
        cam_dir = os.path.join(self.root, "Cameras", cam)
        if not os.path.isdir(cam_dir):
            print(f"[WARN] expected camera dir not found: {cam_dir}")
            return
        for organ_dir in sorted(glob.glob(os.path.join(cam_dir, f"{organ_name}-*"))):
            if not os.path.isdir(organ_dir):
                continue
            for traj_dir in sorted(glob.glob(os.path.join(organ_dir, "TumorfreeTrajectory_*"))):
                if not os.path.isdir(traj_dir):
                    continue
                seq_id = f"{cam}/{os.path.basename(organ_dir)}/{os.path.basename(traj_dir)}"
                frame_paths = sorted(
                    glob.glob(os.path.join(traj_dir, "Frames", "*.jpg"))
                    + glob.glob(os.path.join(traj_dir, "Frames", "*.png"))
                )
                pose_files = glob.glob(os.path.join(traj_dir, "Poses", "*.xlsx"))
                poses_by_frame = self._load_real_camera_poses(pose_files[0]) if pose_files else {}
                if not poses_by_frame:
                    print(f"[WARN] no pose file found under {traj_dir}/Poses -- sequence dropped")
                    continue

                samples = []
                dropped = 0
                for i, fp in enumerate(frame_paths):
                    m = _FRAME_NUMBER_RE.search(os.path.basename(fp))
                    frame_num = int(m.group(1)) if m else None
                    pose = poses_by_frame.get(frame_num) if frame_num is not None else None
                    if pose is None:
                        dropped += 1
                        continue
                    samples.append(FrameSample(fp, pose, None, cam, seq_id, i))
                if dropped:
                    print(f"[WARN] {seq_id}: {dropped}/{len(frame_paths)} frames had no matching "
                          f"pose row (by ImageFrame) and were dropped")
                if samples:
                    sequences[seq_id] = samples

    def _index_unitycam(self, organ_name: str, sequences: dict) -> None:
        organ_dir = os.path.join(self.root, "UnityCam", organ_name)
        if not os.path.isdir(organ_dir):
            print(f"[WARN] expected UnityCam organ dir not found: {organ_dir}")
            return
        seq_id = f"UnityCam/{organ_name}"
        frame_paths = sorted(
            glob.glob(os.path.join(organ_dir, "Frames", "*.png"))
            + glob.glob(os.path.join(organ_dir, "Frames", "*.jpg"))
        )
        depth_dir = os.path.join(organ_dir, "Pixelwise Depths")
        # paired by sorted position, not filename -- Frames/ and Pixelwise Depths/
        # don't share a naming scheme (see module docstring)
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*"))) if os.path.isdir(depth_dir) else []

        pose_dir = os.path.join(organ_dir, "Poses")
        pose_files = glob.glob(os.path.join(pose_dir, "*.csv"))
        poses = self._load_unitycam_poses(pose_files[0]) if pose_files else np.empty((0, 4, 4), dtype=np.float32)
        if len(poses) == 0:
            print(f"[WARN] no pose file found under {pose_dir} -- sequence dropped")
            return

        # UnityCam's pose CSV has no frame-index/name column (see module
        # docstring) -- alignment is positional only. Truncate frames/depths/
        # poses to the shortest of the three rather than letting poses fall
        # back to None mid-sequence: pose supervision in Phase 3 needs pose
        # GT to be non-optional per frame it trains on, unlike depth (which
        # already has a has_depth opt-out below).
        n = min(len(frame_paths), len(poses))
        if n < len(frame_paths):
            print(f"[WARN] {seq_id}: {len(frame_paths)} frames but only {len(poses)} pose rows "
                  f"-- truncating to {n}")
        samples = []
        for i in range(n):
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            samples.append(FrameSample(frame_paths[i], poses[i], depth_path, "UnityCam", seq_id, i))
        if samples:
            sequences[seq_id] = samples

    @staticmethod
    def _load_real_camera_poses(pose_file: str) -> dict[int, np.ndarray]:
        """{ImageFrame -> 4x4 SE(3) matrix}, keyed by the same frame number encoded
        in each frame's filename (frame_NNNNNN.jpg) -- see module docstring."""
        df = pd.read_excel(pose_file)
        t = df[["trans_x", "trans_y", "trans_z"]].to_numpy(dtype=np.float32)
        q = df[["quot_x", "quot_y", "quot_z", "quot_w"]].to_numpy(dtype=np.float32)
        frame_nums = df["ImageFrame"].to_numpy(dtype=np.int64)
        return {
            int(frame_nums[i]): _pose_matrix(t[i], q[i])
            for i in range(len(df))
        }

    @staticmethod
    def _load_unitycam_poses(pose_file: str) -> np.ndarray:
        """(N, 4, 4) SE(3) matrices in file row order -- no alignment key exists,
        caller must pair positionally against sorted frame paths."""
        cols = ["tX", "tY", "tZ", "rX", "rY", "rZ", "rW"]
        df = pd.read_csv(pose_file)
        n_before = len(df)
        # Confirmed on the real file: its last row is a partial write (NaNs in
        # rY/rZ/rW/time(s)), presumably truncated logging -- dropping only
        # trailing NaN rows is positionally safe since it can't shift the
        # index of any earlier row. A NaN row anywhere else WOULD break the
        # positional row<->frame pairing, so flag loudly rather than silently
        # drop mid-sequence.
        nan_rows = df[cols].isna().any(axis=1)
        if nan_rows.any():
            bad_idx = df.index[nan_rows]
            if list(bad_idx) != list(range(n_before - len(bad_idx), n_before)):
                print(f"[WARN] {pose_file}: NaN pose row(s) NOT confined to the tail "
                      f"({list(bad_idx)}) -- dropping them will shift positional "
                      f"frame alignment for everything after the first bad row")
            df = df.dropna(subset=cols)
            print(f"[WARN] {pose_file}: dropped {n_before - len(df)}/{n_before} row(s) "
                  f"with NaN pose values")
        t = df[["tX", "tY", "tZ"]].to_numpy(dtype=np.float32)
        q = df[["rX", "rY", "rZ", "rW"]].to_numpy(dtype=np.float32)
        return np.stack([_pose_matrix(t[i], q[i]) for i in range(len(df))]) if len(df) else np.empty((0, 4, 4), dtype=np.float32)

    def _apply_split(self, train_split: float, val_split: float, seed: int):
        # UnityCam is the ONLY sequence with depth+pose GT (1 of 25 total
        # sequences) -- splitting it like a real-camera sequence (whole-unit,
        # shuffled) risks landing it 100% in one split, leaving val/test with
        # zero quantitative-eval samples. Real-camera sequences keep the
        # original whole-sequence shuffle+slice; UnityCam instead gets sliced
        # positionally by the same fractions *within* its own frame range, so
        # every split gets a share of it. Side effect: up to
        # context_window - 1 frames are lost at each UnityCam split boundary
        # (no window can cross it) -- correct, since it prevents temporal
        # leakage across train/val/test.
        real_keys = [k for k, v in self.sequences.items() if v[0].camera != self.synthetic_cam]
        unity_keys = [k for k, v in self.sequences.items() if v[0].camera == self.synthetic_cam]

        rng = random.Random(seed)
        rng.shuffle(real_keys)
        n = len(real_keys)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        if self.split == "train":
            keep_real = real_keys[:n_train]
        elif self.split == "val":
            keep_real = real_keys[n_train:n_train + n_val]
        else:
            keep_real = real_keys[n_train + n_val:]

        kept = {k: self.sequences[k] for k in keep_real}
        for k in unity_keys:
            samples = self.sequences[k]
            m = len(samples)
            m_train = int(m * train_split)
            m_val = int(m * val_split)
            if self.split == "train":
                kept[k] = samples[:m_train]
            elif self.split == "val":
                kept[k] = samples[m_train:m_train + m_val]
            else:
                kept[k] = samples[m_train + m_val:]
        self.sequences = kept

    def _build_windows(self) -> list[list[FrameSample]]:
        windows = []
        for samples in self.sequences.values():
            for start in range(0, max(1, len(samples) - self.context_window + 1)):
                windows.append(samples[start:start + self.context_window])
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        imgs, depths, poses, has_depth_flags = [], [], [], []
        for s in window:
            img = cv2.imread(s.image_path)
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))

            if s.depth_path is not None:
                d = cv2.imread(s.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                # UnityCam depth PNGs are 8-bit 4-channel (RGBA) with all four
                # channels identical (confirmed via phase3b_explore, 2026-08-13)
                # -- a naive imread/resize keeps 4 channels, silently producing
                # (H,W,4) instead of the documented (T,H,W); take one channel.
                if d.ndim == 3:
                    d = d[..., 0]
                d = cv2.resize(d, self.image_size, interpolation=cv2.INTER_NEAREST)
                depths.append(torch.from_numpy(d))
                has_depth_flags.append(True)
            else:
                depths.append(torch.zeros(self.image_size))
                has_depth_flags.append(False)

            poses.append(torch.from_numpy(s.pose).float() if s.pose is not None else torch.eye(4))

        return {
            "images": torch.stack(imgs),                      # (T, 3, H, W)
            "depths": torch.stack(depths),                     # (T, H, W) -- ignore where has_depth is False
            "poses": torch.stack(poses),                       # (T, 4, 4) -- SE(3) camera pose matrices
            "has_depth": torch.tensor(has_depth_flags),
            "camera": window[0].camera,
            "sequence_id": window[0].sequence_id,
        }

    def flatten_for_enhancement(self) -> list[FrameSample]:
        """Single-frame list for DarkIR-lite training, which doesn't need temporal windows."""
        out = []
        for samples in self.sequences.values():
            out.extend(samples)
        return out
