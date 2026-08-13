"""
Mini-3D-Recon training loop.

Trains on UnityCam only (EndoSLAMStomachDataset(cameras=["UnityCam"])) --
the only source with both depth and pose GT; real cameras have pose GT
only and their coordinate frame/scale relative to UnityCam is unconfirmed
(see PROGRESS.md). Consumes windows directly (unlike darkir_lite/train.py,
which flattens to single frames -- Mini-3D-Recon needs the temporal
context for relative pose).

Checkpoints every `checkpoint_every_steps` (config.yaml) since Kaggle
sessions can die without warning -- and supports --resume so a run that
doesn't fit one session can continue across multiple pushes, mirroring
darkir_lite/train.py's pattern exactly.

Usage (from the repo root, after cloning alongside the DarkIR-lite setup --
see the training notebook's setup cell):
    python -m src.reconstruction.train --config configs/config.yaml \
        --output-dir /kaggle/working/checkpoints [--max-steps N] [--resume PATH]
"""

import argparse
import os

import cv2

# See darkir_lite/train.py for the full rationale -- OpenCV's internal
# thread pool is a known source of deadlocks in PyTorch DataLoader worker
# processes after fork(); disable it before any DataLoader with
# num_workers>0 is created.
cv2.setNumThreads(0)

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.device import select_device
from src.data.endoslam_dataset import EndoSLAMStomachDataset
from src.reconstruction.geometry import relative_pose_from_absolute
from src.reconstruction.loss import depth_absrel, pose_loss, total_loss
from src.reconstruction.model import MiniReconModel


def build_dataloaders(config: dict):
    context_window = config["reconstruction"]["context_window"]
    batch_size = config["reconstruction"]["batch_size"]

    train_ds = EndoSLAMStomachDataset(config, split="train", cameras=["UnityCam"], context_window=context_window)
    val_ds = EndoSLAMStomachDataset(config, split="val", cameras=["UnityCam"], context_window=context_window)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, config: dict, max_batches: int | None = None) -> dict:
    """max_batches caps validation for --max-steps smoke tests, same
    rationale as darkir_lite/train.py's evaluate()."""
    model.eval()
    absrels, rot_errs, trans_errs = [], [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = batch["images"].to(device)
        depths = batch["depths"].to(device)
        has_depth = batch["has_depth"].to(device)
        poses = batch["poses"].to(device)

        pred = model(images)
        absrels.append(depth_absrel(pred["depth"], depths, has_depth).item())

        _, _, rot_err_deg = pose_loss(pred["rotation"], pred["translation"], poses)
        rot_errs.append(rot_err_deg.item())

        gt_rel_translation = relative_pose_from_absolute(poses[:, :-1], poses[:, 1:])[..., :3, 3]
        trans_err = torch.norm(pred["translation"] - gt_rel_translation, dim=-1).mean()
        trans_errs.append(trans_err.item())

    return {
        "val_depth_absrel": sum(absrels) / max(1, len(absrels)),
        "val_rot_err_deg": sum(rot_errs) / max(1, len(rot_errs)),
        "val_trans_err": sum(trans_errs) / max(1, len(trans_errs)),
    }


def save_checkpoint(path, model, optimizer, epoch, global_step, val_metrics: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **(val_metrics or {}),
        },
        path,
    )


def train(config: dict, output_dir: str, resume_from: str | None = None, max_steps: int | None = None):
    device = select_device()
    print(f"device: {device}")

    rcfg = config["reconstruction"]
    train_loader, val_loader = build_dataloaders(config)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = MiniReconModel(pretrained=True, depth_head_channels=rcfg["depth_head_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rcfg["lr"])

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
    for epoch in range(start_epoch, rcfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_components = {"depth_loss": 0.0, "trans_loss": 0.0, "rot_loss": 0.0, "rot_err_deg": 0.0}
        n_batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            batch_gpu = {
                "depths": batch["depths"].to(device),
                "has_depth": batch["has_depth"].to(device),
                "poses": batch["poses"].to(device),
            }

            optimizer.zero_grad()
            pred = model(images)
            loss, components = total_loss(pred, batch_gpu, config)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k in epoch_components:
                epoch_components[k] += components[k]
            n_batches += 1
            global_step += 1

            if global_step % rcfg["checkpoint_every_steps"] == 0:
                path = os.path.join(output_dir, f"step_{global_step}.pt")
                save_checkpoint(path, model, optimizer, epoch, global_step)
                print(f"  checkpoint saved: {path}")
                print(f"  raw components (unweighted): {components}")

            if max_steps and global_step >= max_steps:
                stop_early = True
                break

        avg_loss = epoch_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in epoch_components.items()}

        val_max_batches = 5 if max_steps else None
        val_metrics = evaluate(model, val_loader, device, config, max_batches=val_max_batches)
        print(f"epoch {epoch}: train_loss={avg_loss:.4f} components={avg_components} "
              f"val_depth_absrel={val_metrics['val_depth_absrel']:.4f} "
              f"val_rot_err_deg={val_metrics['val_rot_err_deg']:.2f} "
              f"val_trans_err={val_metrics['val_trans_err']:.4f} (raw UnityCam world-units, not meters)")

        epoch_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_path, model, optimizer, epoch, global_step, val_metrics)

        if stop_early:
            print(f"stopping early at step {global_step} (--max-steps {max_steps})")
            break

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Mini-3D-Recon")
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
