"""
Synthetic low-light degradation, following DarkIR's image formation model:

    y = gamma * (x (*) k) + n

where x is a clean frame, k is a blur kernel (motion/defocus, simulating the
endoscope tip wobbling near tissue), gamma < 1 attenuates brightness
(simulating the tiny single-LED light source), and n is sensor noise.

Why this exists: EndoSLAM's own footage isn't necessarily dark enough to
prove "brightening helps" in a controlled way -- lighting varies by
sub-dataset and camera. Applying a *known, controllable* degradation to
clean frames means Phase 5's before/after comparison is actually measuring
what you degraded, not an uncontrolled mix of real lighting conditions.

Use on: UnityCam (clean synthetic, no real dark issues) and optionally on
HighCam/LowCam frames that are relatively well-lit, to build clean/dark
pairs for supervised DarkIR-lite fine-tuning. Real genuinely-dark HighCam/
LowCam sequences stay untouched -- those are your final "unseen real dark
video" qualitative demo, not training pairs.
"""

import numpy as np
import cv2


def random_motion_blur_kernel(size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simple linear motion blur kernel at a random angle."""
    rng = rng or np.random.default_rng()
    kernel = np.zeros((size, size), dtype=np.float32)
    angle = rng.uniform(0, 180)
    kernel[size // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    kernel /= kernel.sum() + 1e-8
    return kernel


def degrade_frame(
    clean_rgb: np.ndarray,        # HxWx3 float32 in [0,1]
    gamma_range=(0.15, 0.4),
    blur_kernel_range=(3, 9),
    noise_std_range=(0.01, 0.05),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()

    gamma = rng.uniform(*gamma_range)
    k_size = int(rng.integers(blur_kernel_range[0] // 2, blur_kernel_range[1] // 2 + 1)) * 2 + 1
    noise_std = rng.uniform(*noise_std_range)

    kernel = random_motion_blur_kernel(k_size, rng=rng)
    blurred = cv2.filter2D(clean_rgb, -1, kernel)

    darkened = gamma * blurred

    noise = rng.normal(0, noise_std, size=darkened.shape).astype(np.float32)
    noisy = darkened + noise

    return np.clip(noisy, 0.0, 1.0)


def build_paired_dataset_entry(clean_rgb: np.ndarray, config: dict) -> dict:
    """Convenience wrapper matching config.yaml's dark_degradation block."""
    dc = config["dark_degradation"]
    dark = degrade_frame(
        clean_rgb,
        gamma_range=tuple(dc["gamma_range"]),
        blur_kernel_range=tuple(dc["blur_kernel_range"]),
        noise_std_range=tuple(dc["noise_std_range"]),
    )
    return {"clean": clean_rgb, "dark": dark}


if __name__ == "__main__":
    # Quick sanity check on a synthetic gradient image -- run this first in
    # Kaggle before wiring it into the real dataloader, to eyeball that the
    # degradation actually looks like endoscope footage and not just "a
    # dimmed photo."
    h, w = 320, 320
    gradient = np.tile(np.linspace(0.3, 0.9, w, dtype=np.float32), (h, 1))
    fake_clean = np.stack([gradient, gradient * 0.8, gradient * 0.6], axis=-1)
    fake_dark = degrade_frame(fake_clean)
    print("clean range:", fake_clean.min(), fake_clean.max())
    print("dark  range:", fake_dark.min(), fake_dark.max())
