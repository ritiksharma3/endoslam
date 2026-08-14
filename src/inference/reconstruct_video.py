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
from src.fusion.backproject import backproject_depth, transform_points_to_world
from src.fusion.intrinsics import depth_byte_to_unity_units, fov_to_intrinsics
from src.fusion.pointcloud import accumulate_point_cloud, apply_depth_trunc_mask
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
    mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"])
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
) -> o3d.geometry.PointCloud:
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
    contribute points to the fused cloud or the ICP reference window."""
    fcfg = config["fusion"]
    H, W = tuple(config["data"]["image_size"])
    fx, fy, cx, cy = fov_to_intrinsics(fcfg["camera_fov_deg"], (H, W))
    context_window = config["reconstruction"]["context_window"]
    icp_max_corr_dist = icp_max_corr_dist or fcfg["voxel_downsample"] * 10
    icp_voxel_size = fcfg["voxel_downsample"] * 4  # coarser than the final fusion voxel size -- ICP is expensive

    windows = build_windows(frames, context_window)
    n = len(windows)

    darkir_model.eval()
    mini_recon_model.eval()

    all_depths: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_valid: list[bool] = []
    all_rotations: list[torch.Tensor] = []
    all_translations: list[torch.Tensor] = []

    for i in range(n):
        window = windows[i]  # (T,3,H,W)

        # DarkIR is a per-frame model expecting (B,3,H,W) -- the window's T
        # frames go in as its batch dimension directly, not wrapped in an
        # extra unsqueeze(0) (that produced an invalid 5D tensor, see this
        # module's docstring and PROGRESS.md's Phase 5 log).
        enhanced = darkir_model(window.to(device)).clamp(0.0, 1.0)  # (T,3,H,W)

        pred = mini_recon_model(enhanced.unsqueeze(0))  # (1,T,3,H,W) -> dict
        depth = pred["depth"][0]              # (T,H,W)
        rotation = pred["rotation"][0]         # (T-1,3,3)
        translation = pred["translation"][0]   # (T-1,3)
        enhanced_cpu = enhanced.cpu()

        if i < n - 1:
            all_depths.append(depth[0].cpu().numpy())
            all_colors.append(enhanced_cpu[0].permute(1, 2, 0).numpy())
            all_valid.append(frame_quality_score(window[0].permute(1, 2, 0).numpy())[1])
            all_rotations.append(rotation[0].cpu())
            all_translations.append(translation[0].cpu())
        else:
            for t in range(depth.shape[0]):
                all_depths.append(depth[t].cpu().numpy())
                all_colors.append(enhanced_cpu[t].permute(1, 2, 0).numpy())
                all_valid.append(frame_quality_score(window[t].permute(1, 2, 0).numpy())[1])
            for t in range(rotation.shape[0]):
                all_rotations.append(rotation[t].cpu())
                all_translations.append(translation[t].cpu())

    def frame_iter():
        absolute_pose = np.eye(4)
        reference_window: deque = deque(maxlen=icp_window)
        n_fused, n_corrected = 0, 0

        for i, depth_byte in enumerate(all_depths):
            if i > 0:
                relative = np.eye(4)
                relative[:3, :3] = all_rotations[i - 1].numpy()
                relative[:3, 3] = all_translations[i - 1].numpy()
                absolute_pose = absolute_pose @ relative

            depth_units = depth_byte_to_unity_units(depth_byte, fcfg["near_clip"], fcfg["far_clip"])
            mask = apply_depth_trunc_mask(depth_units, fcfg["depth_trunc"])
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
                n_fused += 1
                yield points_world, colors

        print(f"reconstruct_video: fused {n_fused}/{len(all_depths)} frames "
              f"({n_corrected} ICP-corrected, icp_window={icp_window}, "
              f"icp_fitness_threshold={icp_fitness_threshold})")

    return accumulate_point_cloud(frame_iter(), voxel_size=fcfg["voxel_downsample"])


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
