"""
Camera intrinsics and depth-unit conversion for Phase 4 point-cloud fusion.

UnityCam has no confirmed intrinsics anywhere in the EndoSLAM dataset,
paper, or repo -- see PROGRESS.md "Phase 4 camera model -- sourced but
unconfirmed" for the full sourcing. Working values (config.yaml's
fusion.camera_fov_deg/near_clip/far_clip) come from a Camera GameObject in
github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy's Record_scene.unity,
positioned near a physics-simulated "Capsule" object -- plausible, not
100% confirmed to be the exact camera that generated this dataset variant.
Must be empirically validated (does GT-mode backprojection produce a
coherent tube, not a scattered mess?) before trusting predicted-mode output.
"""

import numpy as np


def fov_to_intrinsics(fov_deg: float, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """image_size: (H, W). Returns (fx, fy, cx, cy) for a pinhole camera
    with the given vertical field of view. UnityCam's 320x320 image is
    square, so fx == fy regardless of whether fov_deg is meant as
    vertical or horizontal -- the ambiguity in Unity's serialized
    m_FOVAxisMode is moot for this dataset."""
    H, W = image_size
    fov_rad = np.deg2rad(fov_deg)
    f = (H / 2) / np.tan(fov_rad / 2)
    return f, f, W / 2, H / 2


def depth_byte_to_unity_units(depth_byte: np.ndarray, near: float, far: float) -> np.ndarray:
    """Linear01Depth hypothesis (Unity HDRP's default depth-AOV convention):
    depth_byte in [0, 255] -> near + (depth_byte / 255) * (far - near).
    Apply identically to GT and predicted depth -- neither has ever been
    converted before Phase 4 (Phase 3 trained directly against raw bytes)."""
    return near + (depth_byte / 255.0) * (far - near)
