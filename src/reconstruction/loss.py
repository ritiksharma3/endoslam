"""
Mini-3D-Recon training losses.

Depth: plain masked L1 against raw UnityCam depth pixel values (absolute
units unconfirmed -- see PROGRESS.md "Depth format -- confirmed facts";
this is fine per the official EndoSLAM repo's own median-scaling eval
methodology, which already treats depth as scale-ambiguous).

Pose: translation L1 + a trace-based rotation loss (proportional to
1-cos(theta), smooth near 0 -- avoids arccos's unstable gradient there)
against the GT relative transform computed from the window's absolute
poses via geometry.relative_pose_from_absolute(). A separate no-grad
arccos gives the human-readable degree error for validation logging only.
"""

import torch
import torch.nn.functional as F

from src.reconstruction.geometry import relative_pose_from_absolute


def masked_depth_l1(pred_depth: torch.Tensor, gt_depth: torch.Tensor, has_depth: torch.Tensor) -> torch.Tensor:
    """pred_depth, gt_depth: (B,T,H,W); has_depth: (B,T) bool -> scalar loss.
    Guarded against mask.sum()==0 (shouldn't happen for UnityCam-only
    training, but defensive)."""
    B, T, H, W = pred_depth.shape
    mask = has_depth.float().view(B, T, 1, 1)
    abs_err = (pred_depth - gt_depth).abs() * mask
    denom = (mask.sum() * H * W).clamp(min=1)
    return abs_err.sum() / denom


def depth_absrel(pred_depth: torch.Tensor, gt_depth: torch.Tensor, has_depth: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean(|pred-gt|/gt) over masked pixels -- a lighter per-epoch depth
    metric than full Phase-5 evaluation, mirroring darkir_lite's PSNR/SSIM
    per-epoch logging."""
    B, T, H, W = pred_depth.shape
    mask = has_depth.float().view(B, T, 1, 1)
    rel_err = (pred_depth - gt_depth).abs() / gt_depth.clamp(min=eps) * mask
    denom = (mask.sum() * H * W).clamp(min=1)
    return rel_err.sum() / denom


def pose_loss(pred_rotation: torch.Tensor, pred_translation: torch.Tensor, gt_poses: torch.Tensor):
    """pred_rotation: (B,T-1,3,3), pred_translation: (B,T-1,3),
    gt_poses: (B,T,4,4) absolute SE(3). Returns (trans_loss, rot_loss,
    rot_err_deg) -- rot_err_deg is detached, for logging only, not backprop."""
    gt_rel = relative_pose_from_absolute(gt_poses[:, :-1], gt_poses[:, 1:])  # (B,T-1,4,4)
    gt_rotation = gt_rel[..., :3, :3]
    gt_translation = gt_rel[..., :3, 3]

    trans_loss = F.l1_loss(pred_translation, gt_translation)

    rtr_gt = pred_rotation.transpose(-1, -2) @ gt_rotation
    trace = rtr_gt.diagonal(dim1=-2, dim2=-1).sum(-1)
    rot_loss = (1 - (trace - 1) / 2).mean()

    with torch.no_grad():
        cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
        rot_err_deg = torch.rad2deg(torch.arccos(cos_theta)).mean()

    return trans_loss, rot_loss, rot_err_deg


def total_loss(pred: dict, batch: dict, config: dict):
    """Combines depth + pose terms per config.yaml's reconstruction.*_weight
    keys. Returns (scalar loss for backprop, dict of raw unweighted
    components for logging/tuning)."""
    dcfg = config["reconstruction"]
    d_loss = masked_depth_l1(pred["depth"], batch["depths"], batch["has_depth"])
    trans_loss, rot_loss, rot_err_deg = pose_loss(pred["rotation"], pred["translation"], batch["poses"])

    loss = (
        dcfg["depth_loss_weight"] * d_loss
        + dcfg["pose_loss_weight"] * (trans_loss + dcfg["pose_rotation_weight"] * rot_loss)
    )

    components = {
        "depth_loss": d_loss.item(),
        "trans_loss": trans_loss.item(),
        "rot_loss": rot_loss.item(),
        "rot_err_deg": rot_err_deg.item(),
    }
    return loss, components
