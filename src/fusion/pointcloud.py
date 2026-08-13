"""
Point-cloud accumulation utilities (Open3D).
"""

import numpy as np
import open3d as o3d


def apply_depth_trunc_mask(depth_units: np.ndarray, depth_trunc: float, depth_min: float = 0.0) -> np.ndarray:
    """(H, W) depth in scene units -> (H, W) bool mask, True where depth is
    within (depth_min, depth_trunc]."""
    return (depth_units > depth_min) & (depth_units <= depth_trunc)


def accumulate_point_cloud(frame_iter, voxel_size: float, downsample_every: int = 20) -> o3d.geometry.PointCloud:
    """frame_iter yields (points_world: (N,3) float, colors: (N,3) float in
    [0,1]) per frame. Periodically voxel-downsamples the running
    accumulator rather than only at the end -- a full ~1500-frame UnityCam
    sequence would otherwise hold 100M+ points densely in memory at once."""
    accumulated = o3d.geometry.PointCloud()
    pending_points: list[np.ndarray] = []
    pending_colors: list[np.ndarray] = []

    def flush():
        nonlocal accumulated
        if not pending_points:
            return
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pending_points, axis=0))
        new_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(pending_colors, axis=0))
        accumulated += new_pcd
        accumulated = accumulated.voxel_down_sample(voxel_size)
        pending_points.clear()
        pending_colors.clear()

    for i, (points, colors) in enumerate(frame_iter):
        if len(points) == 0:
            continue
        pending_points.append(points)
        pending_colors.append(colors)
        if (i + 1) % downsample_every == 0:
            flush()
    flush()
    return accumulated
