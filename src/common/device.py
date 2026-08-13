import torch


def select_device() -> str:
    """torch.cuda.is_available() only checks that a CUDA driver is present,
    not that this torch build actually ships kernels for the assigned GPU's
    compute capability. Kaggle's API-pushed kernels default to a P100
    (sm_60), and Kaggle's preinstalled torch build has been observed to
    lack Pascal kernels entirely ("no kernel image is available for
    execution on the device") -- confirmed on this project 2026-08-13.
    Do a real op, not just a presence check, and fall back to CPU rather
    than crash the whole run over an environment mismatch we don't
    control (never pip-install/replace the preinstalled torch build)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda") + torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"CUDA reports available but a test op failed ({e}) -- falling back to CPU")
        return "cpu"
