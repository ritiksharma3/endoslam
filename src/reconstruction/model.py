"""
Mini-3D-Recon: per-frame depth + inter-frame relative pose from a shared
MobileNetV3-Small backbone. Trained on UnityCam only (real cameras have no
depth GT and their pose coordinate frame/scale relative to UnityCam is
unconfirmed -- see PROGRESS.md).

Unlike src/darkir_lite/model.py (which wraps one external pretrained
model's single forward()), this is a genuinely composed architecture
(shared backbone + two new heads) -- a class-based nn.Module is the
natural fit here, a deliberate style difference from darkir_lite/model.py,
not an inconsistency.

Pose is parameterized as consecutive frame-to-frame relative SE(3)
transforms (T-1 per T-frame window) using a 6D continuous rotation
representation (Zhou et al., CVPR 2019) reconstructed via Gram-Schmidt --
see geometry.py. This matches README Phase 4's "pose chaining" language:
composing consecutive relative transforms into a trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.reconstruction.geometry import rotation_6d_to_matrix

_BACKBONE_OUT_CHANNELS = 576  # mobilenet_v3_small's final feature width


class DepthHead(nn.Module):
    """Per-frame dense depth from a (C, h, w) backbone feature map.

    5 upsampling stages take mobilenet_v3_small's 10x10 feature map (320x320
    input, factor-32 stride) back to the full 320x320 input resolution
    (10 * 2**5 == 320); a final explicit interpolate to the target size is
    still used rather than trusting that arithmetic, mirroring DarkIR's own
    pad/crop-back defensiveness for a different input size.

    Final activation is plain ReLU: UnityCam depth GT (see PROGRESS.md
    "Depth format -- confirmed facts") is small non-negative pixel values
    with unconfirmed absolute units, not values needing a bounded/sigmoid
    output.
    """

    def __init__(self, in_channels: int = _BACKBONE_OUT_CHANNELS, mid_channels: int = 64):
        super().__init__()
        widths = [in_channels, 256, 128, mid_channels, mid_channels, mid_channels]
        stages = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ))
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        """feats: (N, C, h, w) -> (N, H, W)"""
        x = feats
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        x = F.relu(x)
        return x.squeeze(1)


class PoseHead(nn.Module):
    """Consecutive-pair relative pose from pooled per-frame embeddings.

    Reuses the shared backbone's pooled feature vector (no separate pose
    encoder) -- keeps the model "mini" per the project's explicit scope
    constraints. Outputs a 9-vector per pair: 6D rotation + 3D translation.
    """

    def __init__(self, in_channels: int = _BACKBONE_OUT_CHANNELS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pooled: (B, T, C) -> rotation (B, T-1, 3, 3), translation (B, T-1, 3)"""
        pairs = torch.cat([pooled[:, :-1], pooled[:, 1:]], dim=-1)  # (B, T-1, 2C)
        out = self.mlp(pairs)  # (B, T-1, 9)
        rotation = rotation_6d_to_matrix(out[..., :6])
        translation = out[..., 6:]
        return rotation, translation


class MiniReconModel(nn.Module):
    def __init__(self, pretrained: bool = True, depth_head_channels: int = 64):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.depth_head = DepthHead(mid_channels=depth_head_channels)
        self.pose_head = PoseHead()

    def forward(self, images: torch.Tensor) -> dict:
        """images: (B, T, 3, H, W) -> {"depth": (B,T,H,W), "rotation": (B,T-1,3,3), "translation": (B,T-1,3)}"""
        B, T, C, H, W = images.shape
        x = images.reshape(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, 576, h, w)

        depth = self.depth_head(feats, out_size=(H, W))  # (B*T, H, W)
        depth = depth.reshape(B, T, H, W)

        pooled = feats.mean(dim=[-1, -2]).reshape(B, T, -1)  # (B, T, 576)
        rotation, translation = self.pose_head(pooled)  # (B,T-1,3,3), (B,T-1,3)

        return {"depth": depth, "rotation": rotation, "translation": translation}
