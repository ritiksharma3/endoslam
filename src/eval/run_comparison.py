"""
Phase 5 orchestration: with/without-DarkIR comparison on the UnityCam test
split. Same role Phase 4's src/fusion/reconstruct.py plays for fusion --
this module owns the model-inference/metrics logic and returns plain data;
the notebook (mirroring reconstruct.py's own division of labor) owns writing
JSON/preview PNGs to /kaggle/working.

Two conditions, both evaluated against the *same* synthetically-degraded
input (config.yaml's dark_degradation block, applied on the fly -- the
dataset itself only ever returns clean frames):
  - "raw_dark_input": degraded frames straight into the trained Mini-3D-Recon
    model (Phase 3 checkpoint).
  - "darkir_lite_enhanced": degraded frames through the trained DarkIR-lite
    checkpoint (Phase 2) first, then into Mini-3D-Recon.

Fairness depends on both conditions seeing byte-identical degraded input --
degrade_window() seeds per *dataset-window index* (not wall-clock/global
random state), so re-degrading the same window for each condition reproduces
the exact same dark frames.

Sequence iteration mirrors reconstruct.py's reconstruct_predicted() exactly
(same "every window contributes index 0, except the last window contributes
all T frames/T-1 pairs" trick, same GT-frame-0 pose anchor convention) so
Phase 4 and Phase 5 treat the dataset's windows identically.
"""

import numpy as np
import torch

from src.data.dark_degradation import degrade_frame
from src.eval.metrics import depth_metrics, trajectory_metrics
from src.reconstruction.geometry import absolute_poses_from_relative


def degrade_window(images: torch.Tensor, base_frame_idx: int, config: dict) -> torch.Tensor:
    """images: (T,3,H,W) clean [0,1] float -> (T,3,H,W) degraded, each frame
    seeded by base_frame_idx + t (the dataset window index) so the same
    window always degrades identically across repeated calls -- required so
    both compared conditions see the same dark input, not two independent
    random degradations."""
    dc = config["dark_degradation"]
    out = []
    for t in range(images.shape[0]):
        rng = np.random.default_rng(seed=base_frame_idx + t)
        clean_np = images[t].permute(1, 2, 0).numpy()
        dark_np = degrade_frame(
            clean_np,
            gamma_range=tuple(dc["gamma_range"]),
            blur_kernel_range=tuple(dc["blur_kernel_range"]),
            noise_std_range=tuple(dc["noise_std_range"]),
            rng=rng,
        )
        out.append(torch.from_numpy(dark_np).permute(2, 0, 1))
    return torch.stack(out)


@torch.no_grad()
def run_condition(
    dataset,
    mini_recon_model: torch.nn.Module,
    config: dict,
    device: str,
    condition: str,
    anchor_pose: np.ndarray,
    darkir_model: torch.nn.Module | None = None,
    n_previews: int = 3,
) -> dict:
    """Runs one condition ('raw_dark_input' or 'darkir_lite_enhanced') over
    the full dataset, returns depth + trajectory metrics against GT plus a
    few example frame triplets for the report."""
    assert condition in ("raw_dark_input", "darkir_lite_enhanced")
    if condition == "darkir_lite_enhanced":
        assert darkir_model is not None, "darkir_lite_enhanced requires darkir_model"

    mini_recon_model.eval()
    if darkir_model is not None:
        darkir_model.eval()

    n = len(dataset)
    pred_depths, gt_depths, has_depth_flags = [], [], []
    rotations, translations = [], []
    previews = []
    preview_stride = max(1, n // n_previews)

    for i in range(n):
        window = dataset[i]
        clean_images = window["images"]  # (T,3,H,W)
        dark_images = degrade_window(clean_images, base_frame_idx=i, config=config)

        enhanced_images = None
        if condition == "darkir_lite_enhanced":
            # DarkIR is a per-frame model expecting (B,3,H,W) -- unlike
            # MiniReconModel below, it has no notion of a temporal window, so
            # the window's T frames go in as DarkIR's batch dimension
            # directly (matches darkir_lite/train.py's own usage), not
            # wrapped in an extra unsqueeze(0) (that produced a 5D tensor,
            # which crashed DarkIR's `_, _, H, W = input.shape` unpack).
            enhanced_images = darkir_model(dark_images.to(device)).clamp(0.0, 1.0).cpu()
            model_input = enhanced_images
        else:
            model_input = dark_images

        pred = mini_recon_model(model_input.unsqueeze(0).to(device))
        depth = pred["depth"][0].cpu()            # (T,H,W)
        rotation = pred["rotation"][0].cpu()       # (T-1,3,3)
        translation = pred["translation"][0].cpu()  # (T-1,3)

        depth_range = range(1) if i < n - 1 else range(depth.shape[0])
        for t in depth_range:
            pred_depths.append(depth[t].numpy())
            gt_depths.append(window["depths"][t].numpy())
            has_depth_flags.append(bool(window["has_depth"][t]))

        pair_range = range(1) if i < n - 1 else range(rotation.shape[0])
        for t in pair_range:
            rotations.append(rotation[t])
            translations.append(translation[t])

        if i % preview_stride == 0 and len(previews) < n_previews:
            previews.append({
                "dark": dark_images[0].permute(1, 2, 0).numpy(),
                "enhanced": enhanced_images[0].permute(1, 2, 0).numpy() if enhanced_images is not None else None,
                "clean": clean_images[0].permute(1, 2, 0).numpy(),
            })

    pred_depths = np.stack(pred_depths)   # (N,H,W)
    gt_depths = np.stack(gt_depths)       # (N,H,W)
    has_depth = np.array(has_depth_flags)  # (N,)
    mask = np.broadcast_to(has_depth[:, None, None], gt_depths.shape)

    d_metrics = depth_metrics(pred_depths, gt_depths, mask)

    rotations = torch.stack(rotations)
    translations = torch.stack(translations)
    pose0 = torch.from_numpy(anchor_pose).float()
    pred_traj = absolute_poses_from_relative(pose0, rotations, translations).numpy()

    gt_traj = _collect_gt_trajectory(dataset)

    assert pred_traj.shape[0] == gt_traj.shape[0], (
        f"pred/gt trajectory length mismatch: {pred_traj.shape[0]} vs {gt_traj.shape[0]}"
    )
    t_metrics = trajectory_metrics(pred_traj, gt_traj)

    return {"depth": d_metrics, "trajectory": t_metrics, "previews": previews}


def _collect_gt_trajectory(dataset) -> np.ndarray:
    """GT absolute poses for every unique frame the windows cover, in the
    same order run_condition() collects predictions -- mirrors the same
    'index 0 per window, all T for the last window' iteration exactly."""
    n = len(dataset)
    poses = []
    for i in range(n):
        window = dataset[i]
        frame_range = range(1) if i < n - 1 else range(window["poses"].shape[0])
        for t in frame_range:
            poses.append(window["poses"][t].numpy())
    return np.stack(poses)


def compare_conditions(
    dataset,
    mini_recon_model: torch.nn.Module,
    darkir_model: torch.nn.Module,
    config: dict,
    device: str,
) -> dict:
    """Runs both configs.yaml eval.compare_conditions entries and returns
    {"raw_dark_input": {...}, "darkir_lite_enhanced": {...}}."""
    anchor_pose = dataset[0]["poses"][0].numpy()  # GT frame 0's absolute pose -- world-frame origin choice only,
                                                   # same convention as Phase 4's reconstruct_predicted()

    results = {}
    for condition in config["eval"]["compare_conditions"]:
        results[condition] = run_condition(
            dataset, mini_recon_model, config, device, condition,
            anchor_pose=anchor_pose,
            darkir_model=darkir_model if condition == "darkir_lite_enhanced" else None,
        )
    return results
