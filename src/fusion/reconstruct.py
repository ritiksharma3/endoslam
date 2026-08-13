"""
Top-level Phase 4 orchestration: turn a dataset (or a trained model's
predictions on it) into a single fused Open3D point cloud.

reconstruct_gt() and reconstruct_predicted() share the same backproject/
transform/accumulate pipeline (backproject.py, pointcloud.py) -- only the
depth/pose *source* differs. reconstruct_gt() must be validated (does it
look like a coherent tube, not a scattered mess?) before reconstruct_
predicted()'s output is trusted as a real Phase 3 model-quality signal
rather than a broken-pipeline artifact -- see PROGRESS.md.

Sequence iteration relies on EndoSLAMStomachDataset.windows already being
stride-1 (window k starts at real-sequence-frame k) -- every consecutive
frame pair (i, i+1) is pair index 0 of the window starting at i, so
iterating `dataset[i]` for i in range(len(dataset)) and taking frame/pair
index 0 from each covers every frame and every consecutive-pair transition
exactly once, except the last window also needs to contribute its
remaining T-1 frames/pairs (the only window covering the sequence's final
frames as anything other than index 0).
"""

import numpy as np
import torch

from src.fusion.backproject import backproject_depth, transform_points_to_world
from src.fusion.intrinsics import depth_byte_to_unity_units, fov_to_intrinsics
from src.fusion.pointcloud import accumulate_point_cloud, apply_depth_trunc_mask
from src.reconstruction.geometry import absolute_poses_from_relative


def reconstruct_gt(dataset, cfg: dict, y_down: bool):
    """Backproject the dataset's own GT depth through its own GT absolute
    pose, per frame -- no model, no chaining (every frame already has a
    valid absolute pose). Validates the FOV/near-far/axis-convention
    hypothesis independent of Phase 3's model quality."""
    fcfg = cfg["fusion"]
    H, W = tuple(cfg["data"]["image_size"])
    fx, fy, cx, cy = fov_to_intrinsics(fcfg["camera_fov_deg"], (H, W))
    n = len(dataset)

    def frame_iter():
        for i in range(n):
            window = dataset[i]
            frame_range = range(1) if i < n - 1 else range(window["images"].shape[0])
            for t in frame_range:
                if not bool(window["has_depth"][t]):
                    continue
                depth_byte = window["depths"][t].numpy()
                depth_units = depth_byte_to_unity_units(depth_byte, fcfg["near_clip"], fcfg["far_clip"])
                mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"])
                points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=y_down)
                pose_abs = window["poses"][t].numpy()
                points_world = transform_points_to_world(points_cam[mask], pose_abs)
                colors = window["images"][t].permute(1, 2, 0).numpy()[mask]
                yield points_world, colors

    return accumulate_point_cloud(frame_iter(), voxel_size=fcfg["voxel_downsample"])


def reconstruct_predicted(dataset, model: torch.nn.Module, cfg: dict, y_down: bool, anchor_pose: np.ndarray):
    """Backproject the trained model's predicted depth through its
    predicted relative poses, chained from anchor_pose. anchor_pose is a
    world-coordinate-frame origin choice only (typically GT frame 0's
    absolute pose) -- everything inside the chain is still 100% model
    output, not supervision."""
    fcfg = cfg["fusion"]
    H, W = tuple(cfg["data"]["image_size"])
    fx, fy, cx, cy = fov_to_intrinsics(fcfg["camera_fov_deg"], (H, W))
    device = next(model.parameters()).device
    n = len(dataset)

    model.eval()
    all_depths: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_rotations: list[torch.Tensor] = []
    all_translations: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(n):
            window = dataset[i]
            images = window["images"].unsqueeze(0).to(device)  # (1,T,3,H,W)
            pred = model(images)
            depth = pred["depth"][0]              # (T,H,W)
            rotation = pred["rotation"][0]         # (T-1,3,3)
            translation = pred["translation"][0]   # (T-1,3)

            if i < n - 1:
                all_depths.append(depth[0].cpu().numpy())
                all_colors.append(window["images"][0].permute(1, 2, 0).numpy())
                all_rotations.append(rotation[0].cpu())
                all_translations.append(translation[0].cpu())
            else:
                for t in range(depth.shape[0]):
                    all_depths.append(depth[t].cpu().numpy())
                    all_colors.append(window["images"][t].permute(1, 2, 0).numpy())
                for t in range(rotation.shape[0]):
                    all_rotations.append(rotation[t].cpu())
                    all_translations.append(translation[t].cpu())

    rotations = torch.stack(all_rotations)        # (N-1,3,3)
    translations = torch.stack(all_translations)  # (N-1,3)
    pose0 = torch.from_numpy(anchor_pose).float()
    absolute_poses = absolute_poses_from_relative(pose0, rotations, translations).numpy()  # (N,4,4)

    def frame_iter():
        for i, depth_byte in enumerate(all_depths):
            depth_units = depth_byte_to_unity_units(depth_byte, fcfg["near_clip"], fcfg["far_clip"])
            mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"])
            points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=y_down)
            points_world = transform_points_to_world(points_cam[mask], absolute_poses[i])
            colors = all_colors[i][mask]
            yield points_world, colors

    return accumulate_point_cloud(frame_iter(), voxel_size=fcfg["voxel_downsample"])
