"""
Rotation/pose utilities shared by Mini-3D-Recon's model, loss, and (later,
Phase 4) pose chaining.

6D rotation representation: Zhou et al., "On the Continuity of Rotation
Representations in Neural Networks" (CVPR 2019). Regressing a raw quaternion
or Euler angles has known discontinuities (sign flips, gimbal lock) that hurt
gradient-based training; the 6D representation plus Gram-Schmidt
orthonormalization is continuous everywhere and is the standard choice for a
network's rotation output.
"""

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) orthonormal rotation matrix, right-handed (det=+1)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def relative_pose_from_absolute(pose_t: torch.Tensor, pose_t1: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4), (..., 4, 4) absolute SE(3) poses -> (..., 4, 4) relative
    transform from t to t+1: inverse(pose_t) @ pose_t1."""
    return torch.linalg.inv(pose_t) @ pose_t1


def absolute_poses_from_relative(pose0: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor) -> torch.Tensor:
    """Inverse of relative_pose_from_absolute: pose_t1 = pose_t @ relative_t.

    pose0: (4, 4) anchor absolute pose. rotations: (N, 3, 3), translations:
    (N, 3) -- relative transforms t -> t+1. Returns (N+1, 4, 4): chain[0] =
    pose0, chain[i+1] = chain[i] @ relative_i. A plain Python loop over N
    matmuls -- N is small per call (one video sequence), not worth batching."""
    relative = torch.eye(4, dtype=pose0.dtype, device=pose0.device).expand(rotations.shape[0], 4, 4).clone()
    relative[:, :3, :3] = rotations
    relative[:, :3, 3] = translations

    chain = [pose0]
    for i in range(rotations.shape[0]):
        chain.append(chain[-1] @ relative[i])
    return torch.stack(chain, dim=0)
