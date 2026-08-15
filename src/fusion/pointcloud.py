"""
Point-cloud accumulation utilities (Open3D).
"""

import numpy as np
import open3d as o3d


DEFAULT_DOWNSAMPLE_EVERY = 20


def apply_depth_trunc_mask(depth_units: np.ndarray, depth_trunc: float, depth_min: float = 0.0) -> np.ndarray:
    """(H, W) depth in scene units -> (H, W) bool mask, True where depth is
    within (depth_min, depth_trunc]."""
    return (depth_units > depth_min) & (depth_units <= depth_trunc)


def accumulate_point_cloud(
    frame_iter,
    voxel_size: float,
    downsample_every: int = DEFAULT_DOWNSAMPLE_EVERY,
    initial: o3d.geometry.PointCloud | None = None,
    start_count: int = 0,
    checkpoint_every: int | None = None,
    on_checkpoint=None,
) -> o3d.geometry.PointCloud:
    """frame_iter yields (points_world: (N,3) float, colors: (N,3) float in
    [0,1]) per frame. Periodically voxel-downsamples the running
    accumulator rather than only at the end -- a full ~1500-frame UnityCam
    sequence would otherwise hold 100M+ points densely in memory at once.
    NOTE: voxel_down_sample is not associative across separately-flushed
    batches (merging pre-downsampled batches doesn't reproduce downsampling
    the raw union once), so `downsample_every` isn't just a memory knob --
    changing it changes the final point cloud, and this function guards
    against silently doing that from an unrelated feature (see below).

    `initial`/`start_count` resume an accumulation that was checkpointed and
    killed partway through a previous call: seed the accumulator with the
    previously-saved cloud, and offset the fused-frame count so
    `checkpoint_every` cadence stays aligned with the caller's own count of
    frames fused so far (not just frames fused in *this* call).

    `checkpoint_every`/`on_checkpoint` let a caller persist the accumulated
    cloud periodically -- e.g. reconstruct_video() uses this to checkpoint
    the fused point cloud in lockstep with its own pose-chain/ICP state.
    Deliberately checked *inside* the `downsample_every` block, never on its
    own: checkpointing must never trigger a flush() that wouldn't have
    happened anyway, or (per the associativity note above) enabling
    checkpointing -- or changing checkpoint_every -- would silently change
    the reconstruction output. The practical effect is that checkpoint
    cadence is rounded up to the nearest multiple of `downsample_every`;
    reconstruct_video() relies on this and rounds the fusion-loop's own
    checkpoint trigger to match, so the two checkpoints (pose state, point
    cloud) stay in sync."""
    accumulated = initial if initial is not None else o3d.geometry.PointCloud()
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
            if checkpoint_every and (start_count + i + 1) % checkpoint_every == 0:
                if on_checkpoint:
                    on_checkpoint(start_count + i, accumulated)
    flush()
    return accumulated
