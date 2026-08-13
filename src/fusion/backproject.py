"""
Pinhole depth backprojection and camera-to-world point transformation.
"""

import numpy as np


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, y_down: bool = True) -> np.ndarray:
    """depth: (H, W) in the same units as fx/fy/cx/cy -> (H, W, 3) camera-
    space points.

    y_down=True: standard CV/OpenCV convention (x right, y down, z
    forward) -- pixel row v increasing downward maps to Y increasing.
    y_down=False: flips Y, for cameras/scenes using a y-up convention
    (e.g. Unity's world space). Which one is correct for this dataset is
    UNCONFIRMED -- resolve empirically, see intrinsics.py's docstring."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    if not y_down:
        Y = -Y
    return np.stack([X, Y, depth], axis=-1)


def transform_points_to_world(points_cam: np.ndarray, pose_abs: np.ndarray) -> np.ndarray:
    """points_cam: (..., 3), pose_abs: (4, 4) absolute SE(3) camera pose
    (camera-to-world) -> world = R @ p + t."""
    R = pose_abs[:3, :3]
    t = pose_abs[:3, 3]
    return points_cam @ R.T + t
