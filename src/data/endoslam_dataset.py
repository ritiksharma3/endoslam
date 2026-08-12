"""
EndoSLAM stomach-subset dataset loader.

TODO before trusting this: the official repo's directory tree is shown as an
image (imgs/datatree.png) rather than documented in text, and the Kaggle
mirror may organize files differently. The FIRST thing to do in Kaggle is:

    import os
    for root, dirs, files in os.walk("/kaggle/input/endoslam"):
        print(root, dirs[:5], files[:5])
        if root.count(os.sep) > 4: break  # don't flood output

...and adjust `_index_sequences()` below to match what you actually see.
Everything else in this file is written to be easy to patch once the real
folder names are known -- the path-guessing logic is isolated in one method.

Expected GT formats (per the EndoSLAM paper):
- Pose: 6-DoF (position + orientation) per frame, time-synchronized, for
  UnityCam, HighCam, LowCam.
- Depth: per-pixel depth maps, UnityCam (synthetic) only. HighCam/LowCam
  have NO depth GT -- don't compute depth loss/metrics against them.
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
    pose: np.ndarray | None      # (4,4) or (6,) depending on how GT is stored -- confirm on first inspection
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
        """
        PATCH THIS after inspecting the real folder layout. Placeholder
        assumes a structure like:
            {root}/{organ}/{camera}/{sequence_name}/frames/*.png (or .jpg)
            {root}/{organ}/{camera}/{sequence_name}/pose.txt
            {root}/{organ}/{camera}/{sequence_name}/depth/*.png   (UnityCam only)
        which mirrors how the paper describes per-sub-dataset organization.
        Adjust glob patterns once confirmed.
        """
        sequences = {}
        for cam in self.cameras:
            cam_dir = os.path.join(self.root, self.organ, cam)
            if not os.path.isdir(cam_dir):
                print(f"[WARN] expected camera dir not found: {cam_dir} -- "
                      f"check config.data.root and run the os.walk snippet in the module docstring")
                continue
            for seq_dir in sorted(glob.glob(os.path.join(cam_dir, "*"))):
                if not os.path.isdir(seq_dir):
                    continue
                seq_id = f"{cam}/{os.path.basename(seq_dir)}"
                frame_paths = sorted(
                    glob.glob(os.path.join(seq_dir, "frames", "*.png"))
                    + glob.glob(os.path.join(seq_dir, "frames", "*.jpg"))
                )
                pose_file = os.path.join(seq_dir, "pose.txt")
                poses = self._load_poses(pose_file) if os.path.isfile(pose_file) else None
                depth_dir = os.path.join(seq_dir, "depth")
                has_depth = cam == "UnityCam" and os.path.isdir(depth_dir)

                samples = []
                for i, fp in enumerate(frame_paths):
                    depth_path = None
                    if has_depth:
                        candidate = os.path.join(depth_dir, os.path.basename(fp))
                        depth_path = candidate if os.path.isfile(candidate) else None
                    pose = poses[i] if poses is not None and i < len(poses) else None
                    samples.append(FrameSample(fp, pose, depth_path, cam, seq_id, i))
                if samples:
                    sequences[seq_id] = samples
        return sequences

    @staticmethod
    def _load_poses(pose_file: str) -> np.ndarray:
        # Placeholder: assumes whitespace-separated 6 or 7 values per line
        # (tx ty tz + quaternion or euler). Confirm the actual column count
        # and convention (camera-to-world vs world-to-camera) against the
        # paper's supplementary material before trusting any pose numbers.
        return np.loadtxt(pose_file)

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
