"""
Phase 5 evaluation metrics: depth (AbsRel/RMSE/delta1) and pose trajectory
(ATE/RPE) -- both against a *relative* GT comparison rather than an
absolute-unit one, since neither depth nor predicted-pose translation is in
confirmed absolute units for this project (see PROGRESS.md "Depth format"
and "Pose format -- confirmed facts").

All functions take/return plain numpy arrays, matching src/fusion's
convention (numpy point clouds) rather than src/reconstruction's torch
convention -- this module runs against already-computed model predictions
(inference output), not inside a training loop.
"""

import numpy as np


def depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray, eps: float = 1e-3) -> dict:
    """pred_depth, gt_depth, mask: same-shape arrays (any dims, typically
    (H,W) or (T,H,W)); mask selects valid GT pixels. Applies per-call
    median-ratio scaling before computing metrics -- matches the official
    EndoSLAM repo's own eval_depth.py methodology (predicted-vs-GT depth is
    treated as scale-ambiguous, not a known absolute unit)."""
    pred = pred_depth[mask].astype(np.float64)
    gt = gt_depth[mask].astype(np.float64)
    gt_safe = gt.clip(min=eps)

    pred_median = np.median(pred)
    scale = np.median(gt) / pred_median if pred_median > eps else 1.0
    scaled_pred = pred * scale

    abs_rel = np.mean(np.abs(scaled_pred - gt) / gt_safe)
    rmse = np.sqrt(np.mean((scaled_pred - gt) ** 2))
    ratio = np.maximum(scaled_pred / gt_safe, gt_safe / np.maximum(scaled_pred, eps))
    delta1 = np.mean(ratio < 1.25)

    return {"AbsRel": float(abs_rel), "RMSE": float(rmse), "delta1": float(delta1), "depth_scale": float(scale)}


def align_umeyama(pred_traj: np.ndarray, gt_traj: np.ndarray, with_scale: bool = True):
    """Rigid(+scale) alignment of a predicted (N,4,4) SE(3) trajectory onto
    GT (N,4,4), via Umeyama (1991) closed-form least-squares -- the standard
    TUM RGB-D benchmark approach. Needed because predicted translation is in
    raw, uncalibrated UnityCam world-units, not directly metric-comparable
    to GT. Returns (aligned_pred_translations (N,3), scale)."""
    pred_t = pred_traj[:, :3, 3].astype(np.float64)
    gt_t = gt_traj[:, :3, 3].astype(np.float64)

    pred_mean = pred_t.mean(axis=0)
    gt_mean = gt_t.mean(axis=0)
    pred_c = pred_t - pred_mean
    gt_c = gt_t - gt_mean

    cov = gt_c.T @ pred_c / len(pred_t)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    if with_scale:
        pred_var = (pred_c ** 2).sum(axis=1).mean()
        scale = (D * np.diag(S)).sum() / pred_var if pred_var > 1e-12 else 1.0
    else:
        scale = 1.0

    aligned = scale * (R @ pred_c.T).T + gt_mean
    return aligned, float(scale)


def trajectory_metrics(pred_traj: np.ndarray, gt_traj: np.ndarray, delta: int = 1, with_scale: bool = True) -> dict:
    """ATE + RPE together, sharing one Umeyama alignment. Two different
    corrections are needed, not one, because a *global rotation+translation*
    offset between the two trajectories cancels out of relative (frame i ->
    frame i+delta) motion automatically -- but a *scale* offset does not:
    scaling every translation by a constant scales every relative-motion
    translation by the same constant, so it survives the inv()-cancellation
    RPE relies on (verified empirically: a synthetic trajectory with a known
    rigid+scale offset gave near-zero RPE rotation error but non-zero RPE
    translation error until scale was corrected first). Predicted pose
    translation here is in raw, uncalibrated UnityCam world-units (see
    PROGRESS.md "Pose format -- confirmed facts"), so this scale mismatch is
    real, not a hypothetical edge case.

    ATE: RMSE of per-frame translation error after full Umeyama
    (rotation+translation+scale) alignment.
    RPE: computed after scale-correcting pred_traj's translations only
    (via the same Umeyama-derived scale) -- rotation offsets need no
    correction, translation offsets need no correction either (only scale
    does), since RPE's inv()-based differencing already cancels any
    consistent global rotation/translation."""
    aligned_pred_t, scale = align_umeyama(pred_traj, gt_traj, with_scale=with_scale)
    gt_t = gt_traj[:, :3, 3].astype(np.float64)
    ate_val = float(np.sqrt(np.mean(np.sum((aligned_pred_t - gt_t) ** 2, axis=1))))

    scaled_pred_traj = pred_traj.astype(np.float64).copy()
    scaled_pred_traj[:, :3, 3] *= scale

    n = scaled_pred_traj.shape[0]
    trans_errs, rot_errs_deg = [], []
    for i in range(n - delta):
        pred_rel = np.linalg.inv(scaled_pred_traj[i]) @ scaled_pred_traj[i + delta]
        gt_rel = np.linalg.inv(gt_traj[i]) @ gt_traj[i + delta]
        err = np.linalg.inv(gt_rel) @ pred_rel

        trans_errs.append(np.linalg.norm(err[:3, 3]))
        cos_theta = np.clip((np.trace(err[:3, :3]) - 1) / 2, -1.0, 1.0)
        rot_errs_deg.append(np.degrees(np.arccos(cos_theta)))

    return {
        "ATE": ate_val,
        "RPE_trans_rmse": float(np.sqrt(np.mean(np.square(trans_errs)))),
        "RPE_rot_rmse_deg": float(np.sqrt(np.mean(np.square(rot_errs_deg)))),
        "trajectory_scale": scale,
    }
