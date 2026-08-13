"""
DarkIR-lite fine-tuning loop.

Feeds EndoSLAMStomachDataset.flatten_for_enhancement() (single frames, not
temporal windows) through dark_degradation.py to build clean/dark pairs,
fine-tunes DarkIR-m (see model.py) with L1 loss only (VGG/FFT perceptual
loss deliberately deferred -- see PROGRESS.md Phase 2 notes), and logs
PSNR/SSIM per epoch via skimage.metrics.

Checkpoints every `checkpoint_every_steps` (config.yaml) since Kaggle
sessions can die without warning -- and supports --resume so a 20-epoch
run that doesn't fit one T4 session can continue across multiple pushes.

Usage (from the repo root, after cloning DarkIR alongside it -- see the
training notebook's setup cell):
    python -m src.darkir_lite.train --config configs/config.yaml \
        --output-dir /kaggle/working/checkpoints [--max-steps N] [--resume PATH]
"""

import argparse
import os

import cv2

# OpenCV keeps its own internal thread pool, which is a known source of
# deadlocks in PyTorch DataLoader worker processes after fork() -- the
# worker inherits a broken thread-pool state. Disabling it here (before any
# DataLoader with num_workers>0 is created) is the standard fix. This path
# was never exercised in Phase 1 (no DataLoader workers there), so it's
# untested outside this module.
cv2.setNumThreads(0)

import numpy as np
import torch
import torch.nn as nn
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import DataLoader, Dataset

from src.common.device import select_device
from src.data.dark_degradation import build_paired_dataset_entry
from src.data.endoslam_dataset import EndoSLAMStomachDataset
from src.darkir_lite.model import build_darkir_lite, freeze_encoder, unfreeze_all


class DarkPairDataset(Dataset):
    """Wraps a flat list of FrameSample into (dark, clean) tensor pairs,
    re-degrading fresh on every access -- dark_degradation.degrade_frame()
    is randomized per call, which acts as data augmentation across epochs
    rather than training against one fixed degradation per frame."""

    def __init__(self, frame_samples, config):
        self.frame_samples = frame_samples
        self.image_size = tuple(config["data"]["image_size"])
        self.config = config

    def __len__(self):
        return len(self.frame_samples)

    def __getitem__(self, idx):
        sample = self.frame_samples[idx]
        img = cv2.imread(sample.image_path)
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        pair = build_paired_dataset_entry(img, self.config)
        clean = torch.from_numpy(pair["clean"]).permute(2, 0, 1).float()
        dark = torch.from_numpy(pair["dark"]).permute(2, 0, 1).float()
        return dark, clean


def build_dataloaders(config: dict):
    all_cameras = [config["data"]["synthetic_cam"]] + config["data"]["real_cams"]
    context_window = config["reconstruction"]["context_window"]
    batch_size = config["darkir_lite"]["batch_size"]

    train_frames = EndoSLAMStomachDataset(
        config, split="train", cameras=all_cameras, context_window=context_window
    ).flatten_for_enhancement()
    val_frames = EndoSLAMStomachDataset(
        config, split="val", cameras=all_cameras, context_window=context_window
    ).flatten_for_enhancement()

    train_loader = DataLoader(
        DarkPairDataset(train_frames, config), batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        DarkPairDataset(val_frames, config), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> tuple[float, float]:
    """max_batches caps how much of the validation set gets evaluated --
    important for --max-steps smoke tests, where running the full val set
    (hundreds of batches) after only a handful of training steps defeats
    the point of a *short* smoke test, especially on a CPU fallback."""
    model.eval()
    psnrs, ssims = [], []
    for i, (dark, clean) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        dark, clean = dark.to(device), clean.to(device)
        pred = model(dark).clamp(0.0, 1.0)
        pred_np = pred.cpu().numpy()
        clean_np = clean.cpu().numpy()
        for p, c in zip(pred_np, clean_np):
            p_hwc = p.transpose(1, 2, 0)
            c_hwc = c.transpose(1, 2, 0)
            psnrs.append(psnr_metric(c_hwc, p_hwc, data_range=1.0))
            ssims.append(ssim_metric(c_hwc, p_hwc, data_range=1.0, channel_axis=2))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def save_checkpoint(path, model, optimizer, epoch, global_step, val_psnr=None, val_ssim=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
        },
        path,
    )


def train(config: dict, output_dir: str, resume_from: str | None = None, max_steps: int | None = None):
    device = select_device()
    print(f"device: {device}")

    dcfg = config["darkir_lite"]
    train_loader, val_loader = build_dataloaders(config)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = build_darkir_lite(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=dcfg["lr"])
    criterion = nn.L1Loss()

    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        print(f"resumed from {resume_from} at epoch {start_epoch}, step {global_step}")

    stop_early = False
    for epoch in range(start_epoch, dcfg["epochs"]):
        if epoch < dcfg["freeze_encoder_epochs"]:
            freeze_encoder(model)
        else:
            unfreeze_all(model)

        model.train()
        epoch_loss = 0.0
        for dark, clean in train_loader:
            dark, clean = dark.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(dark)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % dcfg["checkpoint_every_steps"] == 0:
                path = os.path.join(output_dir, f"step_{global_step}.pt")
                save_checkpoint(path, model, optimizer, epoch, global_step)
                print(f"  checkpoint saved: {path}")

            if max_steps and global_step >= max_steps:
                stop_early = True
                break

        avg_loss = epoch_loss / max(1, len(train_loader))
        # smoke tests (max_steps set) only need a handful of val batches to
        # confirm the eval path works and produces sane (non-NaN) numbers --
        # not the full validation set, which can dwarf the training time itself
        val_max_batches = 5 if max_steps else None
        val_psnr, val_ssim = evaluate(model, val_loader, device, max_batches=val_max_batches)
        print(f"epoch {epoch}: train_loss={avg_loss:.4f} val_psnr={val_psnr:.2f} val_ssim={val_ssim:.4f}")

        epoch_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_path, model, optimizer, epoch, global_step, val_psnr, val_ssim)

        if stop_early:
            print(f"stopping early at step {global_step} (--max-steps {max_steps})")
            break

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DarkIR-lite")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Cap total steps -- for smoke testing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, output_dir=args.output_dir, resume_from=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
