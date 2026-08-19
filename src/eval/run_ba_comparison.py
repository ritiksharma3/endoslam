"""
Phase 6 evaluation: quantifies bundle adjustment's actual benefit against
ground truth, mirroring Phase 5's run_comparison.py A/B pattern but on a
different axis -- bundle_adjust_enabled on/off, not DarkIR on/off.

Every Video1.avi test earlier in Phase 6 had no ground truth at all ("looks
less stacked" was the only signal available). UnityCam's synthetic test
split has real GT pose, so this gives an actual number: does
reconstruct_video()'s bundle-adjustment stage reduce trajectory error
against ground truth, not just change the fused shape.

Reuses reconstruct_video() itself unmodified apart from its return_trajectory
flag (see that function's docstring) -- both conditions run the identical
pipeline (DarkIR-lite -> Mini-3D-Recon -> ICP chain [-> bundle adjustment]),
so DarkIR-lite running on already-clean synthetic frames (a mismatch with
Phase 4/5's DarkIR-free/degraded-input conventions, which this evaluation
doesn't need to match -- it's an isolated A/B on the BA flag alone) doesn't
bias the comparison: both conditions see it equally.
"""

import numpy as np
import torch

from src.eval.metrics import trajectory_metrics
from src.inference.reconstruct_video import reconstruct_video


def _collect_clean_frames_and_gt_trajectory(dataset) -> tuple[torch.Tensor, np.ndarray]:
    """Mirrors run_comparison.py's _collect_gt_trajectory() / reconstruct.py's
    reconstruct_predicted() iteration convention exactly (every window
    contributes frame/pose index 0, except the last window contributes all
    T) -- so frame i of the returned tensor and pose i of the returned GT
    trajectory refer to the same real sequence frame."""
    n = len(dataset)
    frames, poses = [], []
    for i in range(n):
        window = dataset[i]
        frame_range = range(1) if i < n - 1 else range(window["images"].shape[0])
        for t in frame_range:
            frames.append(window["images"][t])
            poses.append(window["poses"][t].numpy())
    return torch.stack(frames), np.stack(poses)


def compare_bundle_adjustment(
    dataset,
    darkir_model: torch.nn.Module,
    mini_recon_model: torch.nn.Module,
    config: dict,
    device: str,
    max_frames: int | None = 200,
) -> dict:
    """Runs reconstruct_video() twice -- bundle_adjust_enabled=False then
    True -- on the same UnityCam test-split frames, and returns
    {"icp_only": {...trajectory_metrics...}, "bundle_adjusted": {...}}.

    max_frames: caps evaluation to the first N test-split frames (default
    200, matching the scale of every real-video Phase 6 test this project
    has run so far) -- the test split's exact size wasn't known in advance
    and reconstruct_video() runs a full ICP-chain-plus-bundle-adjustment
    pass *twice*, so an uncapped run's cost isn't predictable up front. Pass
    None to evaluate the whole split.

    Frame-count mismatch note: reconstruct_video() drops frames failing
    frame_quality_score() (blurry/near-black/blown-out) from the returned
    trajectory -- unlike run_comparison.py's simpler pipeline, which has no
    such filter. Clean synthetic UnityCam frames essentially never trigger
    it in practice, so the two trajectories are expected to align frame-for-
    frame; the assertion below catches it loudly instead of silently
    comparing misaligned poses if that assumption ever breaks."""
    frames, gt_traj = _collect_clean_frames_and_gt_trajectory(dataset)
    if max_frames is not None:
        frames, gt_traj = frames[:max_frames], gt_traj[:max_frames]
    print(f"evaluating bundle adjustment on {frames.shape[0]} frames")

    results = {}
    for label, ba_enabled in [("icp_only", False), ("bundle_adjusted", True)]:
        _, pred_traj = reconstruct_video(
            frames, darkir_model, mini_recon_model, config, device,
            bundle_adjust_enabled=ba_enabled, return_trajectory=True,
        )
        assert pred_traj.shape[0] == gt_traj.shape[0], (
            f"{label}: pred/gt trajectory length mismatch ({pred_traj.shape[0]} vs {gt_traj.shape[0]}) -- "
            f"frame_quality_score() likely dropped some frames; align gt_traj to the fused subset before scoring"
        )
        results[label] = trajectory_metrics(pred_traj, gt_traj)
        print(f"{label}: {results[label]}")

    return results
