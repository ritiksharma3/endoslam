"""
DarkIR-lite (DarkIR-m, width=32) wrapper.

"DarkIR-lite" per this project's decision: config.yaml's width_multiplier=0.5
means 0.5x of the official DarkIR-l (width=64), which lands exactly on the
official DarkIR-m checkpoint (width=32, ~3.31M params) -- real pretrained-
weight fine-tuning, not a from-scratch architecture.

Confirmed against the real repo (github.com/cidautai/DarkIR, MIT license)
and checkpoint on 2026-08-13 -- see PROGRESS.md "DarkIR loading -- confirmed
facts" for the exploration kernel evidence:
- archs.DarkIR.DarkIR(width=32) matches the official DarkIR-m config
  (confirmed via options/test/LOLBlur.yml). Its other constructor defaults
  (middle_blk_num_enc=2, middle_blk_num_dec=2, enc_blk_nums=[1,2,3],
  dec_blk_nums=[3,1,1], dilations=[1,4,9], extra_depth_wise=True) already
  match that config, so width=32 alone is sufficient.
- The only HuggingFace checkpoint (Cidaut/DarkIR/DarkIR_384.pt) is that
  exact variant -- loaded with 0 missing/0 unexpected keys in testing.
- Checkpoint dict key is 'params' (torch.load(path)['params']), no DDP
  "module." prefix needed for a single-GPU (non-distributed) load -- the
  repo's own load_model() helpers add that prefix only because their
  create_model() always wraps in DistributedDataParallel.
- Input convention: plain [0,1] float RGB, (B,3,H,W), no extra mean/std
  normalization -- matches dark_degradation.py's convention already.
  forward() internally pads to a multiple of 8 and crops back to the exact
  input size, so callers never need to pad manually.
- Importing anything from `archs` requires `ptflops` installed first --
  archs/__init__.py does `from ptflops import get_model_complexity_info`
  at module level, unrelated to whether we use FLOPs counting ourselves.

Assumes the official DarkIR repo is already cloned at DARKIR_PATH relative
to the working directory (the training notebook's setup cell does this via
`git clone`, mirroring how this project's own src/ reaches Kaggle) -- this
module only adds it to sys.path and imports from it, it doesn't clone.
"""

import os
import sys

import torch
import torch.nn as nn

DARKIR_PATH = os.environ.get("DARKIR_PATH", "DarkIR_upstream")
HF_REPO = "Cidaut/DarkIR"
HF_CHECKPOINT_FILENAME = "DarkIR_384.pt"


def _ensure_darkir_importable() -> None:
    if DARKIR_PATH not in sys.path:
        sys.path.insert(0, DARKIR_PATH)


def build_darkir_lite(pretrained: bool = True) -> nn.Module:
    """Builds DarkIR-m (width=32), optionally loading the official
    pretrained checkpoint from HuggingFace."""
    _ensure_darkir_importable()
    from archs.DarkIR import DarkIR

    model = DarkIR(width=32)

    if pretrained:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(repo_id=HF_REPO, filename=HF_CHECKPOINT_FILENAME)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["params"])

    return model


def freeze_encoder(model: nn.Module) -> None:
    """Freezes DarkIR's encoder path (intro conv + encoders + downsampling
    convs) for the first `freeze_encoder_epochs` per config.yaml, leaving
    the middle blocks, decoder path, and output conv trainable. DarkIR has
    no single "encoder" submodule -- self.intro/encoders/downs are the
    three that structurally form the downsampling path, the closest match
    to config.yaml's intent (freeze the pretrained feature extractor first,
    let the reconstruction path adapt, then unfreeze everything)."""
    for module in (model.intro, model.encoders, model.downs):
        for p in module.parameters():
            p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
