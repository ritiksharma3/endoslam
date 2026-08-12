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

Pose TODO (deliberately deferred, not blocking Phase 1): real-camera poses
turned out to be one .xlsx per trajectory (e.g.
"low_high_pose_stom2_teste2_low_images.xlsx"), not the assumed pose.txt --
column layout unconfirmed. UnityCam's Poses/ format is also unconfirmed.
Parsing is deferred until a phase that actually needs pose values (Phase 3);
until then FrameSample.pose is always None and __getitem__ zero-fills it.

Depth: per-pixel depth maps exist only under UnityCam/Stomach/Pixelwise Depths/
-- HighCam/LowCam have no depth GT, matching the paper. Depth-to-frame
pairing is by sorted position (index i in Frames <-> index i in Pixelwise
Depths), not matching filenames -- the two dirs don't share a naming scheme.
"""

import os
import glob
import random
from dataclasses import dataclass

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


@dataclass
class FrameSample:
    image_path: str
    pose: np.ndarray | None      # always None for now -- pose parsing deferred, see module docstring
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
                # pose: real cams store one .xlsx per trajectory under Poses/ --
                # parsing deferred, see module docstring. depth: no GT for real cams.
                samples = [
                    FrameSample(fp, None, None, cam, seq_id, i)
                    for i, fp in enumerate(frame_paths)
                ]
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
        samples = []
        for i, fp in enumerate(frame_paths):
            depth_path = depth_paths[i] if i < len(depth_paths) else None
            samples.append(FrameSample(fp, None, depth_path, "UnityCam", seq_id, i))
        if samples:
            sequences[seq_id] = samples

    @staticmethod
    def _load_poses(pose_file: str) -> np.ndarray:
        # Not wired up yet -- real-camera poses are .xlsx (one per trajectory,
        # e.g. "low_high_pose_stom2_teste2_low_images.xlsx"), not the originally
        # assumed pose.txt. Column layout unconfirmed; UnityCam's Poses/ format
        # is also unconfirmed. Deferred until a phase that actually needs pose
        # values -- see module docstring.
        raise NotImplementedError("pose parsing not yet implemented -- see module docstring")

    def _apply_split(self, train_split: float, val_split: float, seed: int):
        seq_ids = sorted(self.sequences.keys())
        rng = random.Random(seed)
        rng.shuffle(seq_ids)
        n = len(seq_ids)
        n_train = int(n * train_split)
        n_val = int(n * val_split)
        if self.split == "train":
            keep = seq_ids[:n_train]
        elif self.split == "val":
            keep = seq_ids[n_train:n_train + n_val]
        else:
            keep = seq_ids[n_train + n_val:]
        self.sequences = {k: v for k, v in self.sequences.items() if k in keep}

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
                d = cv2.resize(d, self.image_size, interpolation=cv2.INTER_NEAREST)
                depths.append(torch.from_numpy(d))
                has_depth_flags.append(True)
            else:
                depths.append(torch.zeros(self.image_size))
                has_depth_flags.append(False)

            poses.append(torch.from_numpy(s.pose).float() if s.pose is not None else torch.zeros(6))

        return {
            "images": torch.stack(imgs),                      # (T, 3, H, W)
            "depths": torch.stack(depths),                     # (T, H, W) -- ignore where has_depth is False
            "poses": torch.stack(poses),                       # (T, 6 or 4x4) -- confirm format
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
