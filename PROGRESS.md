# Progress

Canonical status doc. `README.md` has the project pitch and phase plan;
this file tracks what's actually done. Update this, not the README
checklist, as work progresses.

## Where code runs

- **Local (this machine)**: all authoring — source files, config, the
  notebook itself. No GPU, no dataset here, so nothing here actually
  executes against real data.
- **Kaggle**: GPU execution, dataset access, training, checkpoints,
  evaluation. Primary platform — matches `config.yaml`'s data root and
  the README's Kaggle dataset mirror link. Colab is a fallback only if
  Kaggle's weekly GPU quota becomes a blocker; no Colab-specific code
  exists yet. Note: the actual dataset mount path is *not* the classic
  `/kaggle/input/endoslam` — on this environment it's nested under
  `/kaggle/input/datasets/mcocoz/endoslam`. `find_endoslam_root()` in
  the validation notebook (and any future notebook that reads this
  dataset) resolves it dynamically rather than assuming a fixed path.
- **Round-trip rule**: code reaches Kaggle via `git clone` inside the
  notebook (GitHub is the single source of truth for `src/`), not
  manual upload or copy-paste. Any fix discovered while running on
  Kaggle — e.g. patching `_index_sequences()` once the real folder
  layout is known — must be copied back into local `src/` and committed
  here, not left stranded in a Kaggle notebook edit.
- **GitHub remote**: `https://github.com/ritiksharma3/endoslam.git`
  (`origin`, branch `main`), **public** (changed from private — Kaggle
  kernels have no GitHub credentials, so `git clone` on a private repo
  hangs indefinitely waiting for a credential prompt instead of failing;
  nothing sensitive is in this repo, so public was the simple fix).
- **Kaggle API automation**: `kaggle` CLI is installed locally and
  driven programmatically (`kaggle kernels push/status/output/logs`) —
  pushes `notebooks/` (code_file + `kernel-metadata.json`) as kernel
  `ritiksharma8/endoslam-phase1-validation`, GPU off (not needed for
  Phase 1 validation, saves quota for Phases 2-3). This is how Phase 1
  got validated end-to-end without manual browser steps.

## Phase status

| Phase | Days  | Deliverable | Status |
|-------|-------|-------------|--------|
| 0 | 1 | Repo, env, Kaggle/Colab pipeline, dataset confirmed accessible | Done. Dataset access confirmed via the Kaggle API — `mcocoz/endoslam` (~10.4GB) is real and mounts correctly once the nested path is resolved (see "Where code runs"). |
| 1 | 2-5 | Dataset loader working, synthetic dark-degradation validated | **Done and validated on Kaggle** (kernel `endoslam-phase1-validation`, version 7, `COMPLETE`, 2026-08-13). `_index_sequences()` rewritten to match the real layout (see below). `train_ds`/`val_ds`/`test_ds` built successfully: 17150/1885/2588 windows. Dark-degradation visual check ran without error. Pose parsing intentionally deferred (see TODOs). |
| 2 | 5-10 | DarkIR-lite fine-tuned from pretrained checkpoint, PSNR/SSIM logged | **In progress — DarkIR loading fully confirmed, training code not yet written.** Exploration kernel `endoslam-phase2a-darkir-explore` v3 `COMPLETE` on Kaggle 2026-08-13: model instantiates at 3,321,638 params (matches official DarkIR-m's ~3.31M), checkpoint `DarkIR_384.pt` loads with **0 missing / 0 unexpected keys**, forward pass on a real EndoSLAM frame works end-to-end (`[1,3,320,320]` in and out). See "DarkIR loading — confirmed facts" below for exact API. `src/darkir_lite/` still empty — next is `model.py` + `train.py` against these confirmed facts. |
| 3 | 10-20 | Mini-3D-Recon trained on UnityCam depth+pose GT | Not started. Highest-risk phase per README — cut context window/backbone/epochs first if time runs short, not eval/report. |
| 4 | 20-25 | Pose chaining + depth backprojection -> Open3D point cloud viewer | Not started. |
| 5 | 25-30 | With/without-DarkIR comparison, ATE/RPE + AbsRel/RMSE, report | Not started. |

## Real EndoSLAM layout (confirmed, replaces all earlier guesses)

```
{root}/Cameras/{HighCam,LowCam}/Stomach-{I,II,III}/TumorfreeTrajectory_{1-4}/Frames/*.jpg
{root}/Cameras/{HighCam,LowCam}/Stomach-{I,II,III}/TumorfreeTrajectory_{1-4}/Poses/*.xlsx
{root}/UnityCam/Stomach/Frames/*
{root}/UnityCam/Stomach/Pixelwise Depths/*
{root}/UnityCam/Stomach/Poses/*
```

Real cameras nest under `Cameras/` with 3 per-specimen organ folders x 4
trajectories each (24 sequences total). UnityCam sits at the top level as
a single flat sequence — no specimen/trajectory split. `Cameras/` also has
`MiroCam`/`PillCam` (colon-only, not used by this project) and a
`Calibration/` sibling per camera (not a sequence, correctly excluded by
the `{organ}-*` glob).

## DarkIR loading — confirmed facts (2026-08-13, exploration kernel v3)

Found via local clone inspection of `github.com/cidautai/DarkIR` (MIT) and
empirically verified on Kaggle (see log evidence above):

- **Class**: `archs.DarkIR.DarkIR(nn.Module)`. Constructor:
  `__init__(self, img_channel=3, width=32, middle_blk_num_enc=2,
  middle_blk_num_dec=2, enc_blk_nums=[1,2,3], dec_blk_nums=[3,1,1],
  dilations=[1,4,9], extra_depth_wise=True)` — defaults already match the
  DarkIR-m config, so `DarkIR(width=32)` alone is correct.
- **Import gotcha**: `archs/__init__.py` does `from ptflops import
  get_model_complexity_info` at module level — importing anything from
  `archs` (even just `DarkIR`) requires `ptflops` installed first, or it
  fails with `ModuleNotFoundError`, not an isolated failure.
- **Checkpoint**: HuggingFace `Cidaut/DarkIR` hosts exactly one file,
  `DarkIR_384.pt`, confirmed via `options/test/LOLBlur.yml` to be the
  width=32 variant. Download via
  `huggingface_hub.hf_hub_download(repo_id="Cidaut/DarkIR", filename="DarkIR_384.pt")`.
- **Checkpoint format**: `torch.load(path)['params']` is the raw state
  dict — no "module." DDP prefix needed for a single-GPU (non-distributed)
  load, unlike the repo's own `load_model()` helpers which assume DDP.
- **Input convention**: plain `[0,1]` float RGB, `(B,3,H,W)`, no extra
  mean/std normalization (PIL + `ToTensor()` in their `inference.py`) —
  matches `dark_degradation.py`'s convention already. Output shape equals
  input shape (restoration model, not a classifier/encoder).
- **Their `archs/__init__.py`** also has reusable `load_weights()`,
  `resume_model()`, `save_checkpoint()` helpers (all DDP-oriented) —
  `train.py` can adapt the *pattern* (checkpoint dict shape: `epoch`,
  `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`) but
  should skip the DDP wrapping entirely for a single Kaggle GPU.

## Immediate next steps

1. Implement `src/darkir_lite/model.py` (thin wrapper around the above)
   and `train.py` (training loop against `flatten_for_enhancement()` +
   `dark_degradation.py`, L1 loss, PSNR/SSIM logging, checkpointing).
2. Smoke-test on Kaggle (GPU on, short run) before committing to the full
   20-epoch fine-tune.
3. Before Phase 3 (which needs real pose values): implement pose parsing
   for the `.xlsx` files — see TODOs below.

## Known open TODOs in code (not yet resolved)

- **Pose parsing not implemented.** Real-camera poses are one `.xlsx` per
  trajectory (e.g. `low_high_pose_stom2_teste2_low_images.xlsx`); UnityCam's
  `Poses/` format is also unconfirmed. `FrameSample.pose` is always `None`
  right now and `__getitem__` zero-fills it — fine for Phase 1/2 (frames
  only), **must be fixed before Phase 3** (pose-based training).
- **Split imbalance risk**: `_apply_split()` splits by *sequence*, and
  UnityCam is only 1 sequence out of 25 total (vs. 24 real-camera
  sequences). With an 80/10/10 split it can land entirely in train, val,
  *or* test depending on the shuffle seed — meaning val/test could end up
  with zero depth-GT samples. Not an issue for Phase 1 (structural
  validation only), but worth a deliberate fix (e.g. force UnityCam into
  every split, or split within UnityCam by frame ranges) before Phase 3
  needs UnityCam's depth+pose GT for quantitative eval across all splits.

## Log

- 2026-08-09: Phase 0 scaffold created (config, requirements, README).
- 2026-08-12: Fixed broken directory scaffold (brace-expansion had
  failed in an earlier shell, creating one literal folder instead of
  real subdirs). Wrote `.gitignore`. Session stalled before `git init`.
- 2026-08-12: Resumed. Ran `git init`, found and fixed a `.gitignore`
  bug (`data/` with no leading slash was matching `src/data/` too,
  silently excluding the two source files from the first commit — fixed
  to `/data/`). Made the initial commit (11 files). User chose to keep
  the repo local-only for now rather than push to GitHub immediately.
- 2026-08-12: Added GitHub remote (`ritiksharma3/endoslam`, private),
  pushed `main`. Filled in `phase1_data_validation.ipynb`'s `REPO_URL`
  so it's runnable as-is. Next action is the user's: run it on Kaggle.
- 2026-08-13: Automated the Kaggle run via the Kaggle API instead of a
  manual browser run. Installed `kaggle` locally; user provided an API
  token. First automated run failed fast (`Could not resolve host:
  github.com`) — internet access silently no-ops on kernels unless the
  Kaggle account is phone-verified; user verified. Second run hung 76+
  min at `git clone` with no error — root cause: the repo was private
  and Kaggle has no GitHub credentials, so the clone waits on a
  credential prompt forever instead of failing. Fixed by making the repo
  public (user's choice over a PAT/Kaggle-Secret or dataset-bundle
  alternative). Third+ runs revealed the dataset mount path isn't the
  classic `/kaggle/input/endoslam` but nested under
  `/kaggle/input/datasets/mcocoz/endoslam`, and the real EndoSLAM folder
  layout is completely different from the original guess (see above).
  Rewrote `_index_sequences()` accordingly, deferred pose parsing.
  Kernel `endoslam-phase1-validation` version 7 completed successfully:
  17150/1885/2588 train/val/test windows, dark-degradation check passed.
  **Phase 1 is validated.**
- 2026-08-13: Started Phase 2 planning. Researched DarkIR (CVPR 2025) —
  confirmed live GitHub (`cidautai/DarkIR`, MIT) and HuggingFace
  (`Cidaut/DarkIR`) repos. Decided `width_multiplier: 0.5` = official
  DarkIR-m checkpoint (real pretrained fine-tuning) and L1-only loss to
  start. Fixed `config.yaml`'s checkpoint casing (`cidaut` -> `Cidaut`)
  and documented the width decision inline. No training code written yet
  — next is the exploration notebook to confirm DarkIR's real loading API.
- 2026-08-13: Ran the DarkIR exploration notebook. v1 completed but its
  architecture-loading cells were silently skipped (`archs/__init__.py`
  imports `ptflops` at module level, not installed, broke the whole
  import chain). Cloned the DarkIR repo locally to inspect source
  directly instead of iterating blind on Kaggle — found the exact
  constructor signature, checkpoint key (`'params'`), and the
  `DarkIR_384.pt` = width=32 mapping (via `options/test/LOLBlur.yml`).
  v2 fixed the ptflops install but crashed on a cosmetic bug (`ptflops`
  has no `__version__`). v3 fixed that and completed cleanly: 3,321,638
  params, checkpoint loads with 0 missing/0 unexpected keys, forward pass
  works on both a dummy tensor and a real EndoSLAM frame. See "DarkIR
  loading — confirmed facts" above. **Ready to write `model.py`/`train.py`.**
