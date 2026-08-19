"""
Phase 6: optional global bundle-adjustment (BA) refinement over full-video
reconstruction's keyframe poses.

reconstruct_video()'s existing pose chain (src/inference/reconstruct_video.py)
corrects each new frame only against a *sliding window* of the last
`icp_window` already-fused frames -- documented in PROGRESS.md as the root
cause of full-video drift on real footage ("sliding-window ICP only catches
local revisits ... needs full loop-closure SLAM"). This module adds that
missing global step: joint sparse nonlinear least-squares refinement of every
keyframe's pose, a single shared focal length, and a sparse set of 3D
landmarks, using reprojection error.

Deliberately pure numpy/scipy -- no torch, no Open3D dependency -- so the core
numerics here are testable against synthetic data with no model, dataset, or
Kaggle session involved (see `if __name__ == "__main__"` below).

Landmark tracks come from Open3D's `registration_icp` `correspondence_set`
that `refine_pose_icp()` already computes and currently discards -- not from a
new feature-matching pipeline. Classical 2D keypoint matching (RANSAC
essential-matrix/PnP/homography) was deliberately not used here: this
codebase has no keypoint detector, and endoscopic tissue (low-texture,
specular, non-rigid) is a poor domain for one anyway. Reusing the existing
3D-3D ICP correspondences sidesteps that gap entirely.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# RANSAC depth scale/shift alignment
# ---------------------------------------------------------------------------

def ransac_depth_scale_shift(
    frame_depth: np.ndarray,
    reference_depth: np.ndarray,
    n_iters: int = 200,
    inlier_threshold: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, np.ndarray]:
    """frame_depth, reference_depth: paired 1D depth samples at the same
    overlap pixels -- frame_depth from a new keyframe's raw predicted depth,
    reference_depth implied by the already-fused reconstruction at those same
    pixel locations. Robustly fits `reference ~= scale * frame + shift` via
    minimal-sample (2-point) RANSAC so a few specular-highlight/degenerate
    outlier pixels can't drag a plain least-squares fit off. This is the only
    scale-drift correction in the project that needs no ground truth (unlike
    src/eval/metrics.py's Umeyama/median-ratio corrections, which only run at
    eval time against GT) -- it runs at real inference time, per keyframe,
    before that keyframe's depth ever becomes a BA landmark.

    Returns (scale, shift, inlier_mask); (1.0, 0.0, all-True) if fewer than 2
    valid samples are given (nothing to fit)."""
    rng = rng or np.random.default_rng()
    n = len(frame_depth)
    if n < 2:
        return 1.0, 0.0, np.ones(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        dx = frame_depth[j] - frame_depth[i]
        if abs(dx) < 1e-8:
            continue
        scale = (reference_depth[j] - reference_depth[i]) / dx
        shift = reference_depth[i] - scale * frame_depth[i]
        residual = np.abs(reference_depth - (scale * frame_depth + shift))
        inliers = residual < inlier_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_count < 2:
        return 1.0, 0.0, np.ones(n, dtype=bool)

    # Refit on all inliers via least squares -- more stable than trusting the
    # minimal 2-point sample that happened to win the RANSAC vote.
    A = np.stack([frame_depth[best_inliers], np.ones(best_count)], axis=1)
    b = reference_depth[best_inliers]
    (scale, shift), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(scale), float(shift), best_inliers


# ---------------------------------------------------------------------------
# MAD-based outlier rejection
# ---------------------------------------------------------------------------

def mad_outlier_mask(residuals: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Modified z-score rejection: keeps residuals within `threshold` scaled-
    MADs of the median. 1.4826 is the standard constant making MAD a
    consistent estimator of standard deviation under a normal-residual
    assumption. Used to drop the worst-offending correspondence observations
    before they ever become a BA residual -- cheap defense-in-depth alongside
    (not instead of) the robust loss used inside the solve itself."""
    residuals = np.asarray(residuals, dtype=np.float64)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    if mad < 1e-12:
        return np.ones_like(residuals, dtype=bool)
    modified_z = np.abs(residuals - median) / (1.4826 * mad)
    return modified_z <= threshold


# ---------------------------------------------------------------------------
# Index-tracked voxel downsampling
# ---------------------------------------------------------------------------

def voxel_downsample_with_index(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Returns indices of one representative point per occupied voxel.
    Open3D's own `PointCloud.voxel_down_sample()` discards the mapping back
    to original point indices, which the keyframe correspondence path in
    reconstruct_video.py needs to keep (a downsampled keyframe point must
    still trace back to the pixel it was backprojected from, so BA
    observations can carry real (u, v) pixel coordinates)."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    voxel_idx = np.floor(points / voxel_size).astype(np.int64)
    _, first_indices = np.unique(voxel_idx, axis=0, return_index=True)
    return np.sort(first_indices)


# ---------------------------------------------------------------------------
# Track building from ICP correspondence sets
# ---------------------------------------------------------------------------

def build_tracks_from_correspondences(
    pairwise_correspondences: list[tuple[int, int, np.ndarray]],
    keyframe_points_xyz: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """pairwise_correspondences: list of (i, j, corr) where corr is an (M, 2)
    int array of indices into keyframe_points_xyz[i] (col 0) and
    keyframe_points_xyz[j] (col 1) -- exactly what Open3D's
    `registration_icp(...).correspondence_set` gives for the ICP call that
    aligned new keyframe j against reference keyframe i in
    reconstruct_video.py's existing refine_pose_icp() loop.

    Chains these pairwise links into multi-keyframe tracks via union-find, so
    a point re-observed across several already-fused keyframes becomes one
    landmark with several observations instead of several disconnected
    2-frame landmarks (this is what lets BA see revisits from far earlier in
    the video, not just the last `icp_window` frames).

    Returns (observations, landmark_init):
      observations: (n_obs, 3) int array of (track_id, keyframe_index, point_index)
      landmark_init: (n_tracks, 3) float, mean world position of each track's
        member points under the pre-BA (ICP-chain) poses -- the BA starting guess."""
    parent: dict = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, corr in pairwise_correspondences:
        for pi, pj in corr:
            union((i, int(pi)), (j, int(pj)))

    groups: dict = {}
    for i, j, corr in pairwise_correspondences:
        for pi, pj in corr:
            for node in ((i, int(pi)), (j, int(pj))):
                groups.setdefault(find(node), set()).add(node)

    observations = []
    landmark_init = []
    for track_id, members in enumerate(groups.values()):
        pts = [keyframe_points_xyz[kf][pt] for kf, pt in members]
        landmark_init.append(np.mean(pts, axis=0))
        for kf, pt in members:
            observations.append((track_id, kf, pt))

    if not observations:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)
    return np.array(observations, dtype=np.int64), np.array(landmark_init, dtype=np.float64)


# ---------------------------------------------------------------------------
# Parameter (de)serialization -- camera 0 held fixed at identity (gauge fix,
# matching the project's existing anchor_predicted_to_gt_origin convention).
# Rotation is minimal axis-angle (3 params), not this project's neural-net-
# oriented 6D representation (geometry.py::rotation_6d_to_matrix): that
# representation exists to avoid sign-flip/gimbal discontinuities during
# gradient-based training from scratch, which doesn't apply to local
# iterative refinement from an already-good initial guess -- axis-angle is
# the standard, minimal BA parameterization and keeps the Jacobian block size
# at the conventional 6-per-camera.
# ---------------------------------------------------------------------------

def _pack_params(poses: np.ndarray, focal: float, landmarks: np.ndarray, optimize_focal: bool) -> np.ndarray:
    cam_params = []
    for k in range(1, len(poses)):
        rotvec = Rotation.from_matrix(poses[k][:3, :3]).as_rotvec()
        cam_params.append(np.concatenate([rotvec, poses[k][:3, 3]]))
    cam_block = np.concatenate(cam_params) if cam_params else np.zeros(0)
    focal_part = [focal] if optimize_focal else []
    return np.concatenate([cam_block, focal_part, landmarks.reshape(-1)])


def _unpack_params(
    x: np.ndarray, n_cameras: int, n_landmarks: int, optimize_focal: bool, fixed_focal: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_free = n_cameras - 1
    cam_block = x[: n_free * 6].reshape(n_free, 6)
    if optimize_focal:
        focal = float(x[n_free * 6])
        landmarks = x[n_free * 6 + 1:].reshape(n_landmarks, 3)
    else:
        focal = fixed_focal
        landmarks = x[n_free * 6:].reshape(n_landmarks, 3)

    poses = np.zeros((n_cameras, 4, 4))
    poses[0] = np.eye(4)
    for k in range(n_free):
        poses[k + 1, :3, :3] = Rotation.from_rotvec(cam_block[k, :3]).as_matrix()
        poses[k + 1, :3, 3] = cam_block[k, 3:]
        poses[k + 1, 3, 3] = 1.0
    return poses, focal, landmarks


# ---------------------------------------------------------------------------
# Residuals + sparse Jacobian pattern
# ---------------------------------------------------------------------------

def _residuals(
    x: np.ndarray,
    n_cameras: int,
    n_landmarks: int,
    camera_indices: np.ndarray,
    landmark_indices: np.ndarray,
    pixels: np.ndarray,
    cx: float,
    cy: float,
    target_arc_length: float,
    baseline_weight: float,
    optimize_focal: bool,
    fixed_focal: float,
) -> np.ndarray:
    """2 reprojection residuals per (camera, landmark) observation, plus one
    trailing gauge/scale-anchoring residual (technique #4): constrains the
    optimized trajectory's total inter-keyframe arc length to match the
    pre-BA (ICP-chain) trajectory's own arc length. Pure reprojection-only BA
    is scale-ambiguous for a monocular camera -- without this, the optimizer
    is free to drift the whole reconstruction's scale while still driving
    reprojection error to zero. `baseline_weight` is expected to already be
    scaled by the caller (refine_trajectory) to stay meaningful against
    however many thousands of reprojection residuals exist -- a single
    unscaled gauge residual is numerically negligible in a large problem
    regardless of its own weight.

    optimize_focal/fixed_focal: an earlier real-data evaluation
    (src/eval/run_ba_comparison.py against UnityCam ground truth) found
    jointly optimizing focal length alongside scale let BA find a much
    lower reprojection cost in a materially *less accurate* scale regime
    (trajectory_scale collapsed 0.64->0.12, rotation RPE roughly tripled) --
    focal length and monocular depth scale are ambiguous together in a way
    pose+landmark optimization alone isn't. Defaults to fixed_focal (no
    self-calibration); optimize_focal=True is kept for experimentation, not
    the recommended default."""
    poses, focal, landmarks = _unpack_params(x, n_cameras, n_landmarks, optimize_focal, fixed_focal)

    cams = poses[camera_indices]
    R = cams[:, :3, :3]
    t = cams[:, :3, 3]
    world_pts = landmarks[landmark_indices]

    # world -> camera: pose is camera-to-world (world = R @ p_cam + t, see
    # backproject.py::transform_points_to_world), so p_cam = R^T @ (world - t).
    cam_pts = np.einsum("nij,nj->ni", R.transpose(0, 2, 1), world_pts - t)
    z = np.clip(cam_pts[:, 2], 1e-6, None)
    u = focal * cam_pts[:, 0] / z + cx
    v = focal * cam_pts[:, 1] / z + cy
    reproj = np.stack([u - pixels[:, 0], v - pixels[:, 1]], axis=1).reshape(-1)

    arc_length = np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))
    gauge = np.array([baseline_weight * (arc_length - target_arc_length)])

    return np.concatenate([reproj, gauge])


def _build_sparsity(
    n_free_cams: int, n_landmarks: int, camera_indices: np.ndarray, landmark_indices: np.ndarray,
    optimize_focal: bool,
):
    n_obs = len(camera_indices)
    n_residuals = n_obs * 2 + 1
    focal_cols = 1 if optimize_focal else 0
    n_params = n_free_cams * 6 + focal_cols + n_landmarks * 3
    focal_col = n_free_cams * 6
    landmarks_start = focal_col + focal_cols
    S = lil_matrix((n_residuals, n_params), dtype=bool)

    for obs_i in range(n_obs):
        cam, lm = int(camera_indices[obs_i]), int(landmark_indices[obs_i])
        r0 = obs_i * 2
        if cam > 0:
            c0 = (cam - 1) * 6
            S[r0:r0 + 2, c0:c0 + 6] = True
        if optimize_focal:
            S[r0:r0 + 2, focal_col] = True  # shared focal touches every observation
        lc0 = landmarks_start + lm * 3
        S[r0:r0 + 2, lc0:lc0 + 3] = True

    for cam in range(1, n_free_cams + 1):  # gauge residual: every camera's translation block
        c0 = (cam - 1) * 6
        S[-1, c0:c0 + 6] = True

    return S.tocsr()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def refine_trajectory(
    keyframe_poses: np.ndarray,
    keyframe_points_xyz: list[np.ndarray],
    keyframe_points_px: list[np.ndarray],
    pairwise_correspondences: list[tuple[int, int, np.ndarray]],
    intrinsics_guess: tuple[float, float, float, float],
    cfg: dict,
) -> dict:
    """Refines keyframe_poses ((K,4,4) camera-to-world, from the existing
    ICP-chain), a single shared focal length, and sparse 3D landmarks jointly
    via robust sparse bundle adjustment.

    cfg: the `bundle_adjustment` block of config.yaml. See its keys' comments
    in config.yaml for meaning/defaults.

    Returns {"poses", "focal", "landmarks", "accepted", "diagnostics"}.
    `accepted` is False (poses/focal returned unchanged from the input guess)
    if BA's cost improvement over the pre-BA baseline didn't clear
    `cfg["rollback_min_improvement"]` -- this stage can never make output
    worse than reconstruct_video()'s existing ICP-chain-only baseline
    (generalizes the existing icp_fitness_threshold per-step gate to
    whole-optimization granularity -- technique #10)."""
    fx, fy, cx, cy = intrinsics_guess
    n_cameras = len(keyframe_poses)

    observations, landmark_init = build_tracks_from_correspondences(pairwise_correspondences, keyframe_points_xyz)
    if len(observations) == 0:
        return {
            "poses": keyframe_poses, "focal": fx, "landmarks": np.zeros((0, 3)),
            "accepted": False, "diagnostics": {"reason": "no correspondence tracks"},
        }

    track_ids, camera_indices, point_indices = observations[:, 0], observations[:, 1], observations[:, 2]
    pixels = np.stack([keyframe_points_px[kf][pt] for kf, pt in zip(camera_indices, point_indices)])

    optimize_focal = cfg.get("optimize_focal", False)
    # A single gauge residual is numerically negligible against thousands of
    # reprojection residuals regardless of its raw weight -- scale by
    # sqrt(n_observations) so config.yaml's one default stays meaningful
    # whether BA sees a 40-frame Video1.avi test (~6k observations) or a
    # 155-frame UnityCam eval (~190k observations).
    gauge_weight = cfg["baseline_anchor_weight"] * np.sqrt(len(observations))

    target_arc_length = float(np.sum(np.linalg.norm(np.diff(keyframe_poses[:, :3, 3], axis=0), axis=1)))
    x0 = _pack_params(keyframe_poses, fx, landmark_init, optimize_focal)
    baseline_cost = 0.5 * float(np.sum(_residuals(
        x0, n_cameras, len(landmark_init), camera_indices, track_ids, pixels, cx, cy,
        target_arc_length, gauge_weight, optimize_focal, fx,
    ) ** 2))

    best = {"x": x0, "cost": baseline_cost, "camera_indices": camera_indices, "track_ids": track_ids, "pixels": pixels}
    n_landmarks = len(landmark_init)
    cur_cam_idx, cur_track_ids, cur_pixels = camera_indices, track_ids, pixels

    for _ in range(cfg.get("max_outer_iters", 3)):
        sparsity = _build_sparsity(n_cameras - 1, n_landmarks, cur_cam_idx, cur_track_ids, optimize_focal)
        result = least_squares(
            _residuals, best["x"], jac_sparsity=sparsity, method="trf", tr_solver="lsmr",
            loss=cfg["robust_loss"], f_scale=cfg["f_scale"], max_nfev=cfg["max_nfev"],
            args=(n_cameras, n_landmarks, cur_cam_idx, cur_track_ids, cur_pixels, cx, cy,
                  target_arc_length, gauge_weight, optimize_focal, fx),
        )
        if result.cost >= best["cost"]:
            break  # no improvement this iteration -- keep best-known state (rollback)
        best = {"x": result.x, "cost": result.cost, "camera_indices": cur_cam_idx,
                "track_ids": cur_track_ids, "pixels": cur_pixels}

        # Re-filter observations by current per-observation reprojection error
        # (MAD, technique #11) before the next outer iteration re-solves.
        resid = _residuals(result.x, n_cameras, n_landmarks, cur_cam_idx, cur_track_ids, cur_pixels,
                            cx, cy, target_arc_length, gauge_weight, optimize_focal, fx)
        per_obs_norm = np.linalg.norm(resid[:-1].reshape(-1, 2), axis=1)
        keep = mad_outlier_mask(per_obs_norm, cfg["mad_outlier_threshold"])
        if keep.all() or keep.sum() < 2:
            break
        cur_cam_idx, cur_track_ids, cur_pixels = cur_cam_idx[keep], cur_track_ids[keep], cur_pixels[keep]

    poses, focal, landmarks = _unpack_params(best["x"], n_cameras, n_landmarks, optimize_focal, fx)
    improvement = (baseline_cost - best["cost"]) / max(baseline_cost, 1e-12)
    accepted = improvement > cfg["rollback_min_improvement"]

    return {
        "poses": poses if accepted else keyframe_poses,
        "focal": focal if accepted else fx,
        "landmarks": landmarks,
        "accepted": accepted,
        "diagnostics": {
            "baseline_cost": baseline_cost, "final_cost": best["cost"],
            "relative_improvement": improvement, "n_tracks": n_landmarks, "n_observations": len(observations),
            "optimize_focal": optimize_focal, "gauge_weight": gauge_weight,
        },
    }


if __name__ == "__main__":
    # Synthetic self-check, no model/dataset/Kaggle needed: build a small
    # ground-truth camera+landmark rig, inject pose noise, verify BA recovers
    # poses closer to ground truth than the noisy input -- mirrors this
    # project's reconstruct_gt() empirical-gate pattern (PROGRESS.md), just
    # runnable on a laptop instead of Kaggle.
    rng = np.random.default_rng(0)
    n_cams, n_pts = 6, 40
    true_focal, cx, cy = 300.0, 160.0, 160.0

    true_landmarks = rng.uniform(-0.3, 0.3, size=(n_pts, 3)) + np.array([0, 0, 1.0])
    true_poses = np.zeros((n_cams, 4, 4))
    for k in range(n_cams):
        true_poses[k] = np.eye(4)
        true_poses[k, :3, 3] = [0.05 * k, 0.0, 0.0]

    keyframe_points_xyz, keyframe_points_px, pairwise = [], [], []
    for k in range(n_cams):
        R, t = true_poses[k, :3, :3], true_poses[k, :3, 3]
        cam_pts = (true_landmarks - t) @ R
        px = np.stack([true_focal * cam_pts[:, 0] / cam_pts[:, 2] + cx,
                        true_focal * cam_pts[:, 1] / cam_pts[:, 2] + cy], axis=1)
        world_pts_noisy = true_landmarks + rng.normal(0, 0.01, size=true_landmarks.shape)
        keyframe_points_xyz.append(world_pts_noisy)
        keyframe_points_px.append(px)
        if k > 0:
            pairwise.append((k - 1, k, np.stack([np.arange(n_pts), np.arange(n_pts)], axis=1)))

    noisy_poses = true_poses.copy()
    for k in range(1, n_cams):
        noisy_poses[k, :3, 3] += rng.normal(0, 0.02, size=3)
        noisy_poses[k, :3, :3] = Rotation.from_rotvec(rng.normal(0, 0.05, size=3)).as_matrix() @ noisy_poses[k, :3, :3]

    cfg = {"robust_loss": "soft_l1", "f_scale": 1.0, "max_nfev": 200, "max_outer_iters": 3,
           "mad_outlier_threshold": 3.5, "baseline_anchor_weight": 1.0, "rollback_min_improvement": 0.01}
    out = refine_trajectory(noisy_poses, keyframe_points_xyz, keyframe_points_px, pairwise,
                             (true_focal, true_focal, cx, cy), cfg)

    err_before = np.linalg.norm(noisy_poses[:, :3, 3] - true_poses[:, :3, 3], axis=1).mean()
    err_after = np.linalg.norm(out["poses"][:, :3, 3] - true_poses[:, :3, 3], axis=1).mean()
    print(f"accepted={out['accepted']} mean translation error before={err_before:.4f} after={err_after:.4f}")
    print(f"diagnostics={out['diagnostics']}")
    assert out["accepted"] and err_after < err_before, "BA did not improve synthetic pose error"
    print("OK: synthetic bundle-adjustment self-check passed.")
