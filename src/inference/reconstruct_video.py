"""
End-to-end: a real endoscope video file -> DarkIR-lite enhancement ->
Mini-3D-Recon depth+pose -> fused Open3D point cloud. This is the project's
literal README.md contract ("Input: dark, low-quality endoscope video.
Output: a rotatable 3D point cloud"), finally exercised against an arbitrary
video file rather than EndoSLAMStomachDataset's pre-extracted, pre-windowed
benchmark frames -- reuses the exact same trained models and fusion code
Phases 2-5 already validated (src/darkir_lite/model.py,
src/reconstruction/model.py, src/fusion/{intrinsics,backproject,pointcloud}.py,
src/reconstruction/geometry.py), just with a new frame source.

Two real differences from src/fusion/reconstruct.py's reconstruct_predicted()
(the Phase 4 benchmark version this mirrors):

1. DarkIR-lite always runs first (Phase 5 proved this wins on every
   metric -- see REPORT.md). Each window's frames go into DarkIR-lite as its
   batch dimension directly (DarkIR has no notion of a temporal window,
   unlike Mini-3D-Recon) -- the exact convention fixed in
   src/eval/run_comparison.py after the original version crashed on a 5D
   tensor. Point-cloud colors come from the *enhanced* frame, not the raw
   dark one -- the whole point is a legible reconstruction of genuinely dark
   footage, and the enhanced frame is what a viewer should see.
2. No ground truth exists for an arbitrary video (unlike the benchmark),
   so the pose chain is anchored at the identity matrix (world origin =
   camera pose at frame 0), not a GT pose. Absolute scale of the resulting
   point cloud is therefore whatever Mini-3D-Recon's raw, uncalibrated
   translation units happen to be for this video -- meaningful for
   shape/relative geometry, not real-world distances (same caveat already
   documented for Phase 3/4/5's raw UnityCam world-units).

DOMAIN GAP CAVEAT: Mini-3D-Recon was trained exclusively on synthetic
UnityCam depth+pose ground truth (see PROGRESS.md) -- it has never been
supervised against a real endoscope frame (real HighCam/LowCam sequences
were excluded from Phase 3 training over an unresolved coordinate-frame/
scale mismatch). Running this on real video works end-to-end without
crashing, but depth/pose *quality* on real footage beyond Phase 4's
qualitative spot-check is unverified -- treat the output as a demo, not a
validated measurement, until real footage is evaluated against real GT the
way Phase 5 did for the synthetic domain.

`--best-frame` mode (select_best_frame() + reconstruct_single_frame()):
reconstructs only the single sharpest/best-lit frame instead of chaining
poses across the whole video. Added after a full-video real-footage run
(200 frames) produced a dense blob rather than a clean shape -- expected
per the domain-gap caveat above, but also compounded by pose-chain drift
across many predicted relative poses. A single view has no pose chain to
drift: its own camera frame trivially *is* the world frame. Also streams
the video frame-by-frame rather than buffering it whole, unlike
load_video_frames() -- load_video_frames() holding a multi-minute video
entirely in memory as one float32 tensor was never exercised against
anything longer than the ~200-frame sample and would be several GB for a
real multi-minute video.
"""

import argparse
import os
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import yaml

from src.common.device import select_device
from src.darkir_lite.model import build_darkir_lite
from src.fusion import bundle_adjust
from src.fusion.backproject import backproject_depth, transform_points_to_world
from src.fusion.intrinsics import depth_byte_to_unity_units, fov_to_intrinsics
from src.fusion.pointcloud import (
    DEFAULT_DOWNSAMPLE_EVERY,
    accumulate_point_cloud,
    apply_border_mask,
    apply_depth_trunc_mask,
)
from src.reconstruction.model import MiniReconModel


def load_video_frames(
    video_path: str,
    image_size: tuple[int, int],
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> torch.Tensor:
    """video_path -> (N,3,H,W) float32 [0,1] RGB tensor. Same preprocessing
    EndoSLAMStomachDataset.__getitem__ already applies to every frame
    (cv2.resize -> BGR2RGB -> /255.0), so frames match what the models were
    trained on. frame_stride/max_frames are pragmatic escape hatches for
    long videos -- the whole result is held in memory at once, no
    chunked/streaming processing (same tradeoff run_comparison.py already
    makes for the benchmark's full test split)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    frames = []
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                frame_bgr = cv2.resize(frame_bgr, image_size)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return torch.stack(frames)


def build_windows(frames: torch.Tensor, context_window: int) -> list[torch.Tensor]:
    """(N,3,H,W) -> list of (T,3,H,W) stride-1 windows. Mirrors
    EndoSLAMStomachDataset._build_windows() exactly (same
    range(0, max(1, N - context_window + 1)) formula) so window composition
    matches what Mini-3D-Recon was trained on."""
    n = frames.shape[0]
    return [frames[start:start + context_window] for start in range(0, max(1, n - context_window + 1))]


@torch.no_grad()
def enhance_all_frames(
    frames: torch.Tensor, darkir_model: torch.nn.Module, device: str, batch_size: int = 8
) -> torch.Tensor:
    """DarkIR-lite has no temporal dependency -- it enhances each frame
    independently, regardless of which window it's later grouped into.
    build_windows() below produces stride-1 overlapping windows, so without
    this, reconstruct_video()'s per-window loop would run every physical
    frame through DarkIR-lite up to `context_window` (8) times (once per
    window it appears in) for a single used output. Enhancing every frame
    exactly once here, then slicing windows out of the cached result,
    removes that redundant compute without changing DarkIR-lite's output
    (eval-mode batching doesn't affect per-sample results) or touching
    Mini-3D-Recon's windowed forward pass at all. Returned on CPU -- an
    (N,3,H,W) float32 tensor for a multi-thousand-frame video is sized to
    hold in RAM (same tradeoff load_video_frames() already makes for the
    raw frames), individual windows are moved to `device` on use."""
    darkir_model.eval()
    chunks = []
    for start in range(0, frames.shape[0], batch_size):
        chunk = frames[start:start + batch_size].to(device)
        chunks.append(darkir_model(chunk).clamp(0.0, 1.0).cpu())
    return torch.cat(chunks, dim=0)


def frame_quality_score(frame: np.ndarray, dark_min: float = 0.02, bright_max: float = 0.95) -> tuple[float, bool]:
    """frame: (H,W,3) uint8 BGR straight from cv2.VideoCapture.read() (used
    by select_best_frame()), OR (H,W,3) float32 [0,1] RGB (the format
    reconstruct_video() already works in, used to gate fusion there) --
    accepts either so both call sites share one implementation instead of
    duplicating the heuristic. Returns (sharpness, is_valid).

    sharpness = variance of the Laplacian on grayscale -- standard,
    cheap blur-detection heuristic (higher = sharper). is_valid = mean
    brightness (normalized to [0,1]) falls in (dark_min, bright_max) --
    rejects near-black frames (lens transitions/occlusion, no real signal)
    and blown-out/saturated ones. Deliberately does NOT penalize ordinary
    darkness within that range -- brightening a dark-but-sharp frame is
    exactly DarkIR-lite's job; biasing selection toward "already bright"
    frames would undercut the whole point of this project."""
    if frame.dtype == np.uint8:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean() / 255.0
    is_valid = dark_min < brightness < bright_max
    return float(sharpness), bool(is_valid)


def select_best_frame(
    video_path: str,
    image_size: tuple[int, int],
    frame_stride: int = 1,
) -> tuple[torch.Tensor, int, float]:
    """Streams the video one frame at a time -- unlike load_video_frames(),
    never buffers the whole video in memory (a real concern for a
    multi-minute video: thousands of frames as one float32 tensor could be
    several GB). Scores every frame with frame_quality_score() and keeps
    only the current best frame's data in memory. Returns (best_frame
    (3,H,W) float32 [0,1] RGB -- already resized/converted, model-ready,
    its index among stride-sampled frames, its sharpness score)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    best_score = float("-inf")
    best_frame_rgb: np.ndarray | None = None
    best_index = -1
    idx = 0
    seen = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                score, is_valid = frame_quality_score(frame_bgr)
                if is_valid and score > best_score:
                    best_score = score
                    resized = cv2.resize(frame_bgr, image_size)
                    best_frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    best_index = seen
                seen += 1
            idx += 1
    finally:
        cap.release()

    if best_frame_rgb is None:
        raise ValueError(f"no valid (non-degenerate) frame found in {video_path} -- every frame was "
                          f"too dark/blown-out to pass frame_quality_score's brightness check")
    return torch.from_numpy(best_frame_rgb).permute(2, 0, 1), best_index, best_score


@torch.no_grad()
def reconstruct_single_frame(
    frame: torch.Tensor,
    darkir_model: torch.nn.Module,
    mini_recon_model: torch.nn.Module,
    config: dict,
    device: str,
) -> o3d.geometry.PointCloud:
    """frame: (3,H,W) [0,1] RGB (see select_best_frame). DarkIR-enhances it,
    runs Mini-3D-Recon as a T=1 window (MiniReconModel handles this cleanly
    -- PoseHead's pair-concatenation just produces an empty 0-length
    tensor, no crash, nothing to chain -- confirmed locally before relying
    on it), and backprojects the one frame's depth directly. No pose
    chaining and no anchor-pose ambiguity here: a single view's own camera
    frame trivially *is* the world frame, so there's nothing to drift --
    the whole reason this mode exists is to sidestep reconstruct_video()'s
    multi-frame pose-chain drift on out-of-domain footage."""
    fcfg = config["fusion"]
    H, W = tuple(config["data"]["image_size"])
    fx, fy, cx, cy = fov_to_intrinsics(fcfg["camera_fov_deg"], (H, W))

    darkir_model.eval()
    mini_recon_model.eval()

    enhanced = darkir_model(frame.unsqueeze(0).to(device)).clamp(0.0, 1.0)  # (1,3,H,W) -- DarkIR's batch dim
    pred = mini_recon_model(enhanced.unsqueeze(0))  # (1,1,3,H,W) -> depth (1,1,H,W)
    depth = pred["depth"][0, 0].cpu().numpy()  # (H,W)
    color = enhanced[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)

    depth_units = depth_byte_to_unity_units(depth, fcfg["near_clip"], fcfg["far_clip"])
    raw_color = frame.permute(1, 2, 0).cpu().numpy()  # pre-DarkIR -- see apply_border_mask()'s docstring for why
    mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"]) & apply_border_mask(
        raw_color, fcfg["border_min_brightness"]
    )
    points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=fcfg["depth_axis_y_down"])
    points_world = points_cam[mask]  # identity pose -- camera frame IS world frame, no transform needed
    colors = color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if fcfg["voxel_downsample"]:
        pcd = pcd.voxel_down_sample(fcfg["voxel_downsample"])
    return pcd


def save_preview(pcd: o3d.geometry.PointCloud, path: str, max_points: int = 30000) -> None:
    """Static multi-view matplotlib preview -- doesn't depend on an OpenGL
    context, unlike the interactive o3d.visualization.draw_geometries()
    window (which failed to get a working WGL context in this execution
    environment -- see PROGRESS.md). Same 3-orthographic-view convention
    used by every Kaggle-side notebook's save_preview() helper."""
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j, label) in zip(axes, [(0, 2, "top (X-Z)"), (1, 2, "side (Y-Z)"), (0, 1, "front (X-Y)")]):
        ax.scatter(pts[:, i], pts[:, j], c=cols, s=0.5)
        ax.set_title(label)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def refine_pose_icp(
    points_world_guess: np.ndarray,
    reference_points: np.ndarray,
    max_correspondence_distance: float,
    icp_voxel_size: float,
) -> tuple[np.ndarray, float]:
    """Aligns points_world_guess (a new frame's backprojected points,
    already transformed by the network's predicted initial pose) against
    reference_points (a sliding window of the last N already-placed
    frames' points -- see reconstruct_video()) via point-to-plane ICP.
    Both point sets are voxel-downsampled first, coarser than the final
    fusion voxel size -- ICP is expensive, this keeps per-frame cost
    tractable over a multi-thousand-frame video. Returns (correction
    (4,4) SE(3) to left-multiply onto the initial pose guess, fitness in
    [0,1] -- fraction of source points with a good correspondence, the
    caller's signal for whether to trust this correction or fall back to
    the uncorrected guess when there's genuinely no geometric overlap)."""
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points_world_guess)
    source = source.voxel_down_sample(icp_voxel_size)

    reference = o3d.geometry.PointCloud()
    reference.points = o3d.utility.Vector3dVector(reference_points)
    reference = reference.voxel_down_sample(icp_voxel_size)

    if len(source.points) == 0 or len(reference.points) == 0:
        return np.eye(4), 0.0

    reference.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel_size * 2, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source, reference, max_correspondence_distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation, result.fitness


def collect_keyframe_correspondences(
    keyframe_points: list[np.ndarray],
    keyframe_poses: list[np.ndarray],
    max_correspondence_distance: float,
    fitness_threshold: float,
    loop_closure_radius: float,
    candidate_limit: int,
) -> list[tuple[int, int, np.ndarray]]:
    """For the newest entry in keyframe_points (index len(...)-1), finds ICP
    correspondences against the immediately preceding keyframe (chain
    continuity) plus up to `candidate_limit` earlier keyframes whose
    ICP-chain position is within loop_closure_radius of the new keyframe's
    position -- proximity-based loop-closure candidate search (O(K) distance
    checks per new keyframe, not exhaustive O(K^2) pairwise ICP), so the
    eventual bundle-adjustment stage can see revisits from anywhere earlier
    in the video, not just reconstruct_video()'s existing icp_window.

    Deliberately separate from refine_pose_icp()'s existing per-frame
    correction: that call matches against a downsampled *concatenation* of
    several reference frames, so its correspondence_set indices can't be
    traced back to one specific frame's own points. Each call here compares
    exactly two keyframes' own (already index-tracked, see
    bundle_adjust.voxel_downsample_with_index) point sets, so indices map
    unambiguously back into keyframe_points[i]/[j]. Returns
    [(i, j, correspondence_array), ...]."""
    new_idx = len(keyframe_points) - 1
    if new_idx == 0 or len(keyframe_points[new_idx]) == 0:
        return []

    new_pos = keyframe_poses[new_idx][:3, 3]
    loop_candidates = [
        j for j in range(new_idx - 1)
        if np.linalg.norm(keyframe_poses[j][:3, 3] - new_pos) < loop_closure_radius
    ]
    candidates = [new_idx - 1] + loop_candidates[:candidate_limit]

    out = []
    for j in candidates:
        if len(keyframe_points[j]) == 0:
            continue
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(keyframe_points[new_idx])
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(keyframe_points[j])
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=max_correspondence_distance, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        if result.fitness >= fitness_threshold and len(result.correspondence_set) > 0:
            # registration_icp(source, target, ...).correspondence_set is
            # (source_index, target_index) pairs -- source is keyframe_points[new_idx],
            # target is keyframe_points[j], so the tuple's (i, j) order must match that,
            # not (j, new_idx): build_tracks_from_correspondences indexes col 0 into
            # keyframe_points_xyz[i] and col 1 into keyframe_points_xyz[j].
            out.append((new_idx, j, np.asarray(result.correspondence_set)))
    return out


def correct_keyframe_depth_scale(
    depth_units: np.ndarray,
    mask: np.ndarray,
    absolute_pose: np.ndarray,
    fx: float, fy: float, cx: float, cy: float, y_down: bool,
    reference_points: np.ndarray,
    icp_voxel_size: float,
    icp_max_corr_dist: float,
    ransac_cfg: dict,
) -> tuple[np.ndarray, float, float]:
    """RANSAC depth scale/shift alignment (bundle_adjust.ransac_depth_scale_shift)
    for a new keyframe against the immediately-preceding keyframe's own
    already-placed points, applied BEFORE this keyframe's points become BA
    landmarks. This is the only scale-drift correction in the project that
    needs no ground truth -- src/eval/metrics.py's Umeyama/median-ratio
    corrections only run at eval time against GT. A cheap single-candidate
    ICP (source subsampled to <=2000 points for speed) finds correspondences
    to drive the fit; on failure (no overlap), returns depth unchanged."""
    points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=y_down)
    masked_points_cam = points_cam[mask]
    masked_depth = depth_units[mask]
    n_masked = len(masked_points_cam)
    if n_masked == 0 or len(reference_points) == 0:
        return depth_units, 1.0, 0.0

    stride = max(1, n_masked // 2000)
    sel = np.arange(0, n_masked, stride)
    points_world = transform_points_to_world(masked_points_cam[sel], absolute_pose)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(points_world)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(reference_points)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=icp_voxel_size * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, icp_max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    corr = np.asarray(result.correspondence_set)
    if len(corr) < 2:
        return depth_units, 1.0, 0.0

    raw_depth_samples = masked_depth[sel][corr[:, 0]]
    R, t = absolute_pose[:3, :3], absolute_pose[:3, 3]
    ref_cam = (reference_points[corr[:, 1]] - t) @ R  # world -> this keyframe's camera space
    implied_depth_samples = ref_cam[:, 2]

    scale, shift, _ = bundle_adjust.ransac_depth_scale_shift(
        raw_depth_samples, implied_depth_samples,
        n_iters=ransac_cfg["n_iters"], inlier_threshold=ransac_cfg["inlier_threshold"],
    )
    return np.clip(scale * depth_units + shift, 0.0, None), scale, shift


def rerun_fusion_with_corrected_poses(
    all_depths: list[np.ndarray],
    all_colors: list[np.ndarray],
    all_valid: list[bool],
    all_border_masks: list[np.ndarray],
    fused_original_poses: list[np.ndarray],
    keyframe_fused_index: list[int],
    original_keyframe_poses: np.ndarray,
    ba_keyframe_poses: np.ndarray,
    fcfg: dict,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Re-runs backprojection + fusion (existing, unchanged pointcloud.py/
    backproject.py code) using bundle-adjustment-corrected keyframe poses.
    Every fused frame's corrected pose is obtained by rigidly re-anchoring it
    to its nearest preceding keyframe:
        corrected = ba_pose[k] @ inverse(original_pose[k]) @ original_pose[frame]
    i.e. BA's per-keyframe correction is propagated onto the frames around it
    via the *original* (already fine-grained-ICP-refined) relative motion
    between them, rather than interpolating a new one -- exact for every
    fused frame, keyframe or not, no interpolation approximation needed.

    Returns (pcd, corrected_trajectory) -- the latter is every fused frame's
    corrected absolute pose, (n_fused, 4, 4), used by
    src/eval/run_ba_comparison.py to score bundle adjustment's actual effect
    against ground truth (reconstruct_video()'s point cloud alone has no
    trajectory to compare)."""
    corrections = [
        ba_keyframe_poses[k] @ np.linalg.inv(original_keyframe_poses[k])
        for k in range(len(original_keyframe_poses))
    ]
    corrected_trajectory: list[np.ndarray] = []

    def frame_iter():
        fused_i = 0
        kf_ptr = 0
        for i, valid in enumerate(all_valid):
            if not valid:
                continue
            while kf_ptr + 1 < len(keyframe_fused_index) and keyframe_fused_index[kf_ptr + 1] <= fused_i:
                kf_ptr += 1
            corrected_pose = corrections[kf_ptr] @ fused_original_poses[fused_i]
            corrected_trajectory.append(corrected_pose)

            depth_units = depth_byte_to_unity_units(all_depths[i], fcfg["near_clip"], fcfg["far_clip"])
            mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"]) & all_border_masks[i]
            points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=fcfg["depth_axis_y_down"])
            points_world = transform_points_to_world(points_cam[mask], corrected_pose)
            colors = all_colors[i][mask]
            fused_i += 1
            yield points_world, colors

    pcd = accumulate_point_cloud(frame_iter(), voxel_size=fcfg["voxel_downsample"])
    return pcd, np.stack(corrected_trajectory)


def save_reconstruction_checkpoint(
    checkpoint_path: str,
    frame_index: int,
    absolute_pose: np.ndarray,
    reference_window: deque,
    n_fused: int,
    n_corrected: int,
    accumulated: o3d.geometry.PointCloud,
) -> None:
    """Saves reconstruct_video()'s complete fusion-loop state -- the pose
    chain, the ICP reference window, the fused/corrected counters, AND the
    accumulated point cloud -- as one file, so a killed multi-hour Kaggle
    run loses nothing past the last checkpoint instead of the entire run.
    Deliberately one file, not two (an earlier version split the point
    cloud into its own .ply written from a separate call site): a crash
    between two separate writes leaves them permanently inconsistent --
    whichever order they're written in, resuming from the newer file paired
    with the older one either re-fuses already-fused frames (duplicate
    points) or skips a range entirely (a gap) -- and there's no way to
    detect the mismatch after the fact from the files themselves. Writing
    to a temp path and os.replace()-ing it into place makes the single
    checkpoint file's update atomic: a crash mid-write leaves the *previous*
    checkpoint fully intact, never a half-written one."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    tmp_path = checkpoint_path + ".tmp"
    torch.save(
        {
            "frame_index": frame_index,
            "absolute_pose": absolute_pose,
            "reference_window": list(reference_window),
            "n_fused": n_fused,
            "n_corrected": n_corrected,
            "points": np.asarray(accumulated.points),
            "colors": np.asarray(accumulated.colors),
        },
        tmp_path,
    )
    os.replace(tmp_path, checkpoint_path)


def load_reconstruction_checkpoint(checkpoint_path: str, icp_window: int):
    """Returns (frame_index, absolute_pose, reference_window, n_fused,
    n_corrected, accumulated_pcd) -- everything reconstruct_video() needs
    to resume its fusion loop right after `frame_index` instead of from
    frame 0."""
    state = torch.load(checkpoint_path, weights_only=False)
    reference_window = deque(state["reference_window"], maxlen=icp_window)
    accumulated = o3d.geometry.PointCloud()
    accumulated.points = o3d.utility.Vector3dVector(state["points"])
    accumulated.colors = o3d.utility.Vector3dVector(state["colors"])
    return (
        state["frame_index"], state["absolute_pose"], reference_window,
        state["n_fused"], state["n_corrected"], accumulated,
    )


@torch.no_grad()
def reconstruct_video(
    frames: torch.Tensor,
    darkir_model: torch.nn.Module,
    mini_recon_model: torch.nn.Module,
    config: dict,
    device: str,
    icp_window: int = 15,
    icp_max_corr_dist: float | None = None,
    icp_fitness_threshold: float = 0.3,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
    bundle_adjust_enabled: bool = False,
    return_trajectory: bool = False,
) -> o3d.geometry.PointCloud | tuple[o3d.geometry.PointCloud, np.ndarray]:
    """frames: (N,3,H,W) clean-or-dark [0,1] RGB (see load_video_frames).
    Runs DarkIR-lite -> Mini-3D-Recon per window (unchanged from before),
    then fuses **incrementally**: each frame's pose starts as a guess
    chained from the *previous refined* pose (not a one-shot batch chain
    computed up front), then gets corrected by ICP against a sliding
    window of the last `icp_window` already-placed frames' points
    (refine_pose_icp()) before being added to the point cloud. This
    targets two real problems a naive uncorrected chain has on real
    footage: drift compounding every step with nothing to correct it, and
    no mechanism to recognize when the camera revisits the same tissue
    (e.g. back-and-forth motion) -- a naive chain just places the revisit
    wherever the already-drifted trajectory says to, producing duplicate/
    offset surfaces. ICP against recent frames pulls a genuine revisit
    back into alignment instead. This is geometric self-consistency, not
    ground-truth accuracy -- see this module's docstring for the real
    limits (still monocular depth, still the documented domain gap, and
    sliding-window ICP only catches *local* revisits, not a revisit from
    far earlier in the video -- that needs full loop-closure SLAM).

    Frames that fail frame_quality_score() (blurry/near-black/blown-out --
    scored on the *raw* pre-DarkIR frame, since enhancement can make a
    genuinely-degenerate frame look artificially sharp) still advance the
    pose chain, so the next frame's initial guess stays sane, but don't
    contribute points to the fused cloud or the ICP reference window.

    checkpoint_path/checkpoint_every/resume: periodically (every
    `checkpoint_every` fused frames, rounded up to the nearest multiple of
    accumulate_point_cloud's DEFAULT_DOWNSAMPLE_EVERY -- see that
    function's docstring for why) persist the fusion loop's state (pose
    chain, ICP reference window, accumulated point cloud) to
    checkpoint_path so a killed/interrupted run can continue with --resume
    instead of restarting from frame 0. Only the fusion loop is
    checkpointed -- the forward-pass phase above (DarkIR-lite +
    Mini-3D-Recon) always reruns in full on resume, which is fine given how
    much cheaper Fix 2's dedupe made it.

    bundle_adjust_enabled (Phase 6, default off, every other parameter/path
    above is unaffected by it): after the incremental fusion above completes,
    optionally runs src/fusion/bundle_adjust.py's global sparse bundle
    adjustment over a sparse set of keyframes (every
    config.yaml bundle_adjustment.keyframe_stride-th fused frame) to correct
    the drift the sliding-window ICP above structurally cannot -- it only
    ever sees the last `icp_window` frames, never a revisit from far earlier
    in the video (see this module's docstring / PROGRESS.md). If bundle
    adjustment's cost improvement clears bundle_adjustment.
    rollback_min_improvement, the whole point cloud is re-fused (see
    rerun_fusion_with_corrected_poses()) with the corrected poses; otherwise
    the ICP-chain-only result above is returned completely unchanged.
    NOTE: not resume-aware -- a resumed run's bundle adjustment only sees
    keyframes captured during *this* invocation (a warning is printed).

    return_trajectory: if True, also returns every fused frame's final
    absolute pose as (n_fused, 4, 4) -- whichever pose actually produced the
    returned point cloud (bundle-adjustment-corrected if it ran and was
    accepted, the plain ICP-chain pose otherwise). The point cloud alone has
    no per-frame trajectory to compare against ground truth, which
    src/eval/run_ba_comparison.py needs to score bundle adjustment's actual
    effect on pose accuracy, not just its effect on the fused shape."""
    fcfg = config["fusion"]
    ba_cfg = config.get("bundle_adjustment", {})  # only required when bundle_adjust_enabled=True
    H, W = tuple(config["data"]["image_size"])
    fx, fy, cx, cy = fov_to_intrinsics(fcfg["camera_fov_deg"], (H, W))
    context_window = config["reconstruction"]["context_window"]
    icp_max_corr_dist = icp_max_corr_dist or fcfg["voxel_downsample"] * 10
    icp_voxel_size = fcfg["voxel_downsample"] * 4  # coarser than the final fusion voxel size -- ICP is expensive

    if checkpoint_every:
        # must land on the same cadence accumulate_point_cloud() actually
        # checkpoints at (see its docstring) so the pose-state and
        # point-cloud checkpoints always describe the same fused-frame
        # count -- otherwise a resume would double-fuse or drop frames.
        checkpoint_every = -(-checkpoint_every // DEFAULT_DOWNSAMPLE_EVERY) * DEFAULT_DOWNSAMPLE_EVERY

    resume_state = None
    if resume and checkpoint_path and os.path.isfile(checkpoint_path):
        resume_state = load_reconstruction_checkpoint(checkpoint_path, icp_window)
        print(f"reconstruct_video: resuming from checkpoint at frame {resume_state[0]} "
              f"({resume_state[3]} frames already fused)")
        if bundle_adjust_enabled:
            print("reconstruct_video: WARNING -- bundle_adjust_enabled with --resume only sees "
                  "keyframes captured during this invocation, not the pre-resume portion of the video")

    # DarkIR-enhance every physical frame exactly once (see
    # enhance_all_frames()'s docstring) instead of re-enhancing each frame
    # up to `context_window` times as it recurs across overlapping windows.
    enhanced_frames = enhance_all_frames(
        frames, darkir_model, device, batch_size=config["darkir_lite"]["batch_size"]
    )
    windows = build_windows(enhanced_frames, context_window)
    n = len(windows)

    mini_recon_model.eval()

    all_depths: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_valid: list[bool] = []
    all_border_masks: list[np.ndarray] = []
    all_rotations: list[torch.Tensor] = []
    all_translations: list[torch.Tensor] = []

    for i in range(n):
        enhanced_cpu = windows[i]  # (T,3,H,W), already DarkIR-enhanced, on CPU

        pred = mini_recon_model(enhanced_cpu.unsqueeze(0).to(device))  # (1,T,3,H,W) -> dict
        depth = pred["depth"][0]              # (T,H,W)
        rotation = pred["rotation"][0]         # (T-1,3,3)
        translation = pred["translation"][0]   # (T-1,3)

        if i < n - 1:
            all_depths.append(depth[0].cpu().numpy())
            all_colors.append(enhanced_cpu[0].permute(1, 2, 0).numpy())
            # quality-scored/border-masked on the raw pre-DarkIR frame (see
            # frame_quality_score()'s and apply_border_mask()'s docstrings) --
            # window i's position 0 is always raw frame i under stride-1 windowing.
            raw_i = frames[i].permute(1, 2, 0).numpy()
            all_valid.append(frame_quality_score(raw_i)[1])
            all_border_masks.append(apply_border_mask(raw_i, fcfg["border_min_brightness"]))
            all_rotations.append(rotation[0].cpu())
            all_translations.append(translation[0].cpu())
        else:
            for t in range(depth.shape[0]):
                all_depths.append(depth[t].cpu().numpy())
                all_colors.append(enhanced_cpu[t].permute(1, 2, 0).numpy())
                raw_it = frames[i + t].permute(1, 2, 0).numpy()
                all_valid.append(frame_quality_score(raw_it)[1])
                all_border_masks.append(apply_border_mask(raw_it, fcfg["border_min_brightness"]))
            for t in range(rotation.shape[0]):
                all_rotations.append(rotation[t].cpu())
                all_translations.append(translation[t].cpu())

    # Updated right before every yield below, read back by the
    # accumulate_point_cloud() on_checkpoint callback at the bottom of this
    # function -- checkpointing both the pose-chain state and the point
    # cloud from that single call site (instead of frame_iter() saving its
    # own state independently) means a crash can only ever leave both
    # checkpoint files describing the last *fully processed* fused frame,
    # never the pose state ahead of the point cloud it should match.
    last_yielded_state: dict = {}

    # Phase 6 bundle-adjustment bookkeeping (unused, zero overhead, unless
    # bundle_adjust_enabled). pixel_grid matches backproject_depth's own
    # (u, v) meshgrid convention, so pixel_grid[mask] aligns index-for-index
    # with points_cam[mask] below -- needed to give BA's landmark
    # observations real (u, v) pixel coordinates, not just 3D positions.
    pixel_grid = np.stack(np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)), axis=-1)
    keyframe_points: list[np.ndarray] = []
    keyframe_pixels: list[np.ndarray] = []
    keyframe_poses: list[np.ndarray] = []
    keyframe_fused_index: list[int] = []
    pairwise_correspondences: list[tuple[int, int, np.ndarray]] = []
    fused_original_poses: list[np.ndarray] = []

    def frame_iter():
        if resume_state:
            start_i = resume_state[0] + 1
            absolute_pose = resume_state[1]
            reference_window: deque = resume_state[2]
            n_fused, n_corrected = resume_state[3], resume_state[4]
        else:
            start_i = 0
            absolute_pose = np.eye(4)
            reference_window: deque = deque(maxlen=icp_window)
            n_fused, n_corrected = 0, 0

        for i in range(start_i, len(all_depths)):
            depth_byte = all_depths[i]
            if i > 0:
                relative = np.eye(4)
                relative[:3, :3] = all_rotations[i - 1].numpy()
                relative[:3, 3] = all_translations[i - 1].numpy()
                absolute_pose = absolute_pose @ relative

            depth_units = depth_byte_to_unity_units(depth_byte, fcfg["near_clip"], fcfg["far_clip"])
            mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"]) & all_border_masks[i]
            points_cam = backproject_depth(depth_units, fx, fy, cx, cy, y_down=fcfg["depth_axis_y_down"])
            points_cam_masked = points_cam[mask]
            colors = all_colors[i][mask]

            points_world = transform_points_to_world(points_cam_masked, absolute_pose)

            if len(reference_window) > 0 and len(points_world) > 0:
                reference_points = np.concatenate(list(reference_window), axis=0)
                correction, fitness = refine_pose_icp(
                    points_world, reference_points, icp_max_corr_dist, icp_voxel_size
                )
                if fitness >= icp_fitness_threshold:
                    absolute_pose = correction @ absolute_pose
                    points_world = transform_points_to_world(points_cam_masked, absolute_pose)
                    n_corrected += 1

            if all_valid[i]:
                reference_window.append(points_world)
                fused_original_poses.append(absolute_pose.copy())

                if bundle_adjust_enabled and (len(fused_original_poses) - 1) % ba_cfg["keyframe_stride"] == 0:
                    is_first_keyframe = len(keyframe_points) == 0
                    kf_points_world = points_world
                    if not is_first_keyframe and ba_cfg["ransac_scale_shift"]["enabled"]:
                        corrected_depth, _, _ = correct_keyframe_depth_scale(
                            depth_units, mask, absolute_pose, fx, fy, cx, cy, fcfg["depth_axis_y_down"],
                            keyframe_points[-1], icp_voxel_size, icp_max_corr_dist, ba_cfg["ransac_scale_shift"],
                        )
                        corrected_cam = backproject_depth(corrected_depth, fx, fy, cx, cy, y_down=fcfg["depth_axis_y_down"])
                        kf_points_world = transform_points_to_world(corrected_cam[mask], absolute_pose)
                    idx = bundle_adjust.voxel_downsample_with_index(kf_points_world, icp_voxel_size)
                    keyframe_points.append(kf_points_world[idx])
                    keyframe_pixels.append(pixel_grid[mask][idx])
                    keyframe_poses.append(absolute_pose.copy())
                    keyframe_fused_index.append(len(fused_original_poses) - 1)
                    pairwise_correspondences.extend(collect_keyframe_correspondences(
                        keyframe_points, keyframe_poses, icp_max_corr_dist, icp_fitness_threshold,
                        ba_cfg["loop_closure_radius"], ba_cfg["loop_closure_candidates"],
                    ))

                n_fused += 1
                if checkpoint_path:
                    last_yielded_state.update(
                        i=i, absolute_pose=absolute_pose.copy(),
                        reference_window=deque(reference_window, maxlen=icp_window),
                        n_fused=n_fused, n_corrected=n_corrected,
                    )
                yield points_world, colors

        print(f"reconstruct_video: fused {n_fused}/{len(all_depths)} frames "
              f"({n_corrected} ICP-corrected, icp_window={icp_window}, "
              f"icp_fitness_threshold={icp_fitness_threshold})")

    def on_checkpoint(_i: int, pcd: o3d.geometry.PointCloud) -> None:
        save_reconstruction_checkpoint(
            checkpoint_path,
            last_yielded_state["i"], last_yielded_state["absolute_pose"],
            last_yielded_state["reference_window"],
            last_yielded_state["n_fused"], last_yielded_state["n_corrected"],
            pcd,
        )

    pcd = accumulate_point_cloud(
        frame_iter(),
        voxel_size=fcfg["voxel_downsample"],
        initial=resume_state[5] if resume_state else None,
        start_count=resume_state[3] if resume_state else 0,
        checkpoint_every=checkpoint_every if checkpoint_path else None,
        on_checkpoint=on_checkpoint if checkpoint_path else None,
    )

    trajectory = np.stack(fused_original_poses) if fused_original_poses else np.zeros((0, 4, 4))

    if bundle_adjust_enabled and len(keyframe_points) >= 2:
        original_keyframe_poses = np.stack(keyframe_poses)
        ba_result = bundle_adjust.refine_trajectory(
            original_keyframe_poses, keyframe_points, keyframe_pixels,
            pairwise_correspondences, (fx, fy, cx, cy), ba_cfg,
        )
        print(f"reconstruct_video: bundle adjustment over {len(keyframe_points)} keyframes -- "
              f"accepted={ba_result['accepted']} diagnostics={ba_result['diagnostics']}")
        if ba_result["accepted"]:
            pcd, trajectory = rerun_fusion_with_corrected_poses(
                all_depths, all_colors, all_valid, all_border_masks, fused_original_poses, keyframe_fused_index,
                original_keyframe_poses, ba_result["poses"], fcfg, fx, fy, cx, cy,
            )
    elif bundle_adjust_enabled:
        print(f"reconstruct_video: bundle adjustment skipped -- only {len(keyframe_points)} keyframe(s) captured")

    return (pcd, trajectory) if return_trajectory else pcd


def main():
    parser = argparse.ArgumentParser(description="Reconstruct a 3D point cloud from an endoscope video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--darkir-checkpoint", required=True)
    parser.add_argument("--mini-recon-checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="outputs/reconstruction.ply")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--best-frame", action="store_true",
                         help="reconstruct only the single sharpest/best-lit frame (no pose chaining, "
                              "avoids multi-frame drift on out-of-domain footage) instead of the whole video")
    parser.add_argument("--icp-window", type=int, default=15,
                         help="full-video mode only: number of recent already-placed frames used as the ICP "
                              "reference when correcting each new frame's pose")
    parser.add_argument("--icp-max-corr-dist", type=float, default=None,
                         help="full-video mode only: ICP max correspondence distance, in config.yaml's fusion "
                              "scene units; defaults to 10x fusion.voxel_downsample")
    parser.add_argument("--icp-fitness-threshold", type=float, default=0.3,
                         help="full-video mode only: minimum ICP fitness to trust a pose correction; below this, "
                              "fall back to the network's raw predicted pose")
    parser.add_argument("--checkpoint-path", default=None,
                         help="full-video mode only: file to periodically save fusion-loop state to (pose "
                              "chain, ICP window, accumulated point cloud, written atomically as one file), "
                              "so a killed run can continue with --resume instead of restarting from frame 0")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                         help="full-video mode only: save a checkpoint every N fused frames "
                              "(requires --checkpoint-path); config.yaml's reconstruction.checkpoint_every_steps "
                              "is used if not set")
    parser.add_argument("--resume", action="store_true",
                         help="full-video mode only: resume from --checkpoint-path if it exists")
    parser.add_argument("--bundle-adjust", action="store_true",
                         help="full-video mode only: after ICP-chain fusion completes, run global sparse bundle "
                              "adjustment (Phase 6, src/fusion/bundle_adjust.py) over keyframes to correct drift "
                              "the sliding-window ICP above can't (no loop closure past --icp-window frames). "
                              "Default off; only replaces the output if it clears config.yaml's "
                              "bundle_adjustment.rollback_min_improvement, so this can never make output worse "
                              "than the existing ICP-chain-only result. Tunables live in config.yaml's "
                              "bundle_adjustment block, not as CLI flags -- this is a new feature, config-only "
                              "until real runs show a need for per-invocation overrides (same reasoning "
                              "--icp-window etc. only became flags after active Phase 4 tuning)")
    parser.add_argument("--no-view", action="store_true", help="skip popping the interactive Open3D window")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = select_device()
    print(f"device: {device}")
    image_size = tuple(config["data"]["image_size"])

    darkir_model = build_darkir_lite(pretrained=False).to(device)
    darkir_checkpoint = torch.load(args.darkir_checkpoint, map_location=device, weights_only=False)
    darkir_model.load_state_dict(darkir_checkpoint["model_state_dict"])

    mini_recon_model = MiniReconModel(
        pretrained=False, depth_head_channels=config["reconstruction"]["depth_head_channels"]
    ).to(device)
    mini_recon_checkpoint = torch.load(args.mini_recon_checkpoint, map_location=device, weights_only=False)
    mini_recon_model.load_state_dict(mini_recon_checkpoint["model_state_dict"])

    if args.best_frame:
        frame, frame_index, score = select_best_frame(args.video, image_size, frame_stride=args.frame_stride)
        print(f"selected frame {frame_index} (sharpness score {score:.1f}) from {args.video}")
        pcd = reconstruct_single_frame(frame, darkir_model, mini_recon_model, config, device)
    else:
        frames = load_video_frames(args.video, image_size, frame_stride=args.frame_stride, max_frames=args.max_frames)
        print(f"loaded {frames.shape[0]} frames from {args.video}")
        pcd = reconstruct_video(
            frames, darkir_model, mini_recon_model, config, device,
            icp_window=args.icp_window,
            icp_max_corr_dist=args.icp_max_corr_dist,
            icp_fitness_threshold=args.icp_fitness_threshold,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every or config["reconstruction"]["checkpoint_every_steps"],
            resume=args.resume,
            bundle_adjust_enabled=args.bundle_adjust,
        )
    print(f"reconstructed point cloud: {len(pcd.points)} points")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"saved: {args.output}")

    preview_path = os.path.splitext(args.output)[0] + "_preview.png"
    save_preview(pcd, preview_path)
    print(f"saved preview: {preview_path}")

    if not args.no_view:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
