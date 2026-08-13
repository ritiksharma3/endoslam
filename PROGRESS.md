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
| 2 | 5-10 | DarkIR-lite fine-tuned from pretrained checkpoint, PSNR/SSIM logged | **Done.** Full 20-epoch fine-tune completed in a single Kaggle session on GPU (kernel `endoslam-phase2-darkir-training`, `COMPLETE`, 2026-08-13) — no resume cycle needed. Final checkpoint `epoch_19.pt`: `global_step=43240` (= 2162 steps/epoch x 20, exact), **val_psnr=32.40, val_ssim=0.9226** — up from the 20-step smoke test's 23.68/0.772. Checkpoint left in Kaggle kernel output only (not committed to the repo); re-fetch via `kaggle kernels output ritiksharma8/endoslam-phase2-darkir-training` when Phase 3/4 need the trained weights. |
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

## GPU compatibility issue (2026-08-13, blocks the real training run)

Kaggle kernels pushed via the API with `enable_gpu: true` default to a
**Tesla P100** (compute capability 6.0, Pascal) — and Kaggle's preinstalled
torch build (2.10.0) ships **no Pascal kernels at all**. `torch.cuda.is_available()`
returns `True` (a driver is present) but every real op raises `CUDA error:
no kernel image is available for execution on the device`. This is a known,
documented Kaggle/PyTorch issue, not a bug in our code — see
[Kaggle product feedback #664303](https://www.kaggle.com/product-feedback/664303).

Tried `kaggle kernels push --accelerator nvidiaTeslaT4` to request a T4
instead — Kaggle still assigned a P100 (the exact accelerator enum strings
aren't documented, and/or T4 capacity wasn't available). Rather than keep
guessing, `train.py::_select_device()` does a real tiny op on `cuda` before
trusting it and falls back to CPU on failure — this is what let the smoke
test complete instead of crashing, but CPU was only viable for a 20-step
smoke test, not the real run.

**Fixed (2026-08-13, user has ~28h P100 quota and wants it used, not spent
chasing T4).** PyTorch dropped Pascal (sm_60) support in 2.8; versions
2.4-2.7 still ship sm_60 kernels; DarkIR's own repo pins exactly
`torch==2.5.1`/`torchvision==0.20.1` — squarely in the compatible range.
The training notebook now reinstalls that exact pin inside the Kaggle
kernel (`notebooks/phase2_training`'s setup cell) — a deliberate, narrow
exception to "never touch preinstalled torch," justified because the
preinstalled build is already broken for the hardware Kaggle assigns here.
First attempt used `--index-url` (wrong — replaces PyPI entirely, broke
resolution of `nvidia-cudnn-cu12`, a PyPI-hosted transitive dep, so the
reinstall silently failed and the original broken torch stayed active);
fixed to `--extra-index-url`. **Confirmed working** on kernel
`endoslam-phase2-darkir-training` v7: `torch: 2.5.1+cu121`, `device: cuda`,
`GPU FIX CONFIRMED: real CUDA op succeeded`. 20 steps + partial val ran in
~14.5s on GPU vs ~10 min on the CPU fallback (40x+ speedup).

**Checkpoint persistence across sessions**: a full 20-epoch run (~2162
steps/epoch, extrapolated ~20-25 min/epoch → ~7-8h total) likely won't fit
one Kaggle session. The notebook checks for
`checkpoints_resume/darkir_lite_latest.pt` after cloning and passes it to
`train.py --resume` automatically if present — after any session that
doesn't reach epoch 19, the latest `epoch_*.pt` gets downloaded and
committed to that path, then the notebook re-pushed to continue.

## Pose format — confirmed facts (2026-08-13, `phase3a_pose_explore` kernel v1)

Nobody had ever opened a real pose file before this kernel — both formats
were guesses. Real facts, from `kaggle kernels output
ritiksharma8/endoslam-phase3a-pose-explore`:

- **Real-camera poses** (`Cameras/{cam}/Stomach-*/TumorfreeTrajectory_*/Poses/*.xlsx`,
  one file per trajectory, confirmed — not per-frame): single sheet
  (`Sheet1`), columns `Unnamed: 0, ImageFrame, Pose_Index, trans_x, trans_y,
  trans_z, quot_x, quot_y, quot_z, quot_w`. Quaternion (`quot_*`) +
  translation (`trans_*`) representation — manually verified unit quaternion
  (norm ≈ 1.000007 on row 0: `0.08517² + 0.98372² + (-0.108621)² +
  0.115055² ≈ 1`). Translation magnitudes ~0.18–0.45 — consistent with
  meters at endoscope scale (matches `fusion.depth_trunc: 0.15`).
  **Row count does not reliably equal frame count**: HighCam/Stomach-I/
  Trajectory_1 had 1094 pose rows vs. 1092 frames (mismatch), while
  LowCam/Stomach-II/Trajectory_2 (30 vs 30) and HighCam/Stomach-III/
  Trajectory_3 (749 vs 749) matched exactly. **`ImageFrame` does not start
  at 0** (min=60 for Trajectory_1, up to max=1770) — it's the original
  video frame index, not a 0-based row position. Frame filenames confirmed
  via `kaggle datasets files` as `frame_{N:06d}.jpg` (e.g.
  `frame_000741.jpg`) — **`ImageFrame` matches the zero-padded number in
  the filename directly**, so real-camera pose/frame alignment must be by
  parsing that number from the filename and joining on `ImageFrame`, NOT
  by positional/row order (positional order happened to work for 2 of 3
  sampled trajectories only by coincidence of no dropped frames).
- **UnityCam poses**: NOT `.xlsx` — a single
  `UnityCam/Stomach/Poses/stomach_position_rotation.csv` for the whole
  sequence. Columns: `tX, tY, tZ, rX, rY, rZ, rW, time(s)`. Also
  quaternion+translation, but **no frame-index/name column at all** — only
  a `time(s)` column incrementing by ~0.0333334s (~30fps). Translation
  magnitudes are much larger (e.g. `tX=0.66, tY=8.98, tZ=-3.13`) than
  real-camera poses — different scale/coordinate system (Unity world
  units, not endoscope-scale meters) — **do not assume the same units or
  coordinate convention as real-camera poses without an explicit
  conversion**. Row count vs. frame count: 1544 rows vs. 1548 frames (and
  1548 depth maps) — a small mismatch, and with no alignment column
  available, positional truncate-to-min is the only option here (unlike
  real cams, which have `ImageFrame` to align by).

## Immediate next steps

1. Phase 3: implement `_load_poses()` against the confirmed formats above
   — real-camera loader joins on `ImageFrame` (parsed from both the xlsx
   column and the `frame_NNNNNN.jpg` filename), UnityCam loader truncates
   positionally to `min(frame_count, pose_row_count)` since there's no
   alignment key. **Done** — see the Log entry below.
2. Write the Mini-3D-Recon model itself (backbone/pose-head/training loop)
   — deliberately out of scope for the loader work above.

## Known open TODOs in code (not yet resolved)

- **Pose parsing**: implemented — `_load_real_camera_poses()` and
  `_load_unitycam_poses()` in `src/data/endoslam_dataset.py`, validated
  locally against the real downloaded sample files (not just the kernel
  log's `.describe()` summary). One more real-data wrinkle found only by
  testing against the actual file: the UnityCam CSV's last row is a
  partial write (`NaN` in `rY/rZ/rW/time(s)`) — `scipy.spatial.transform.
  Rotation.from_quat` raises on a zero/NaN-derived quaternion, so the
  loader now drops NaN rows explicitly, with a loud warning if a dropped
  row isn't confined to the tail (which would silently shift positional
  frame alignment for every row after it — didn't happen on the real
  file, but the loader guards for it since a future dataset version
  could).
- **Coordinate-frame/handedness between real-camera and UnityCam poses is
  still unconfirmed.** Both use quaternion+translation, but UnityCam's
  translation scale is clearly different (Unity world units vs. real-cam's
  apparent meters) and Unity itself is left-handed/Y-up — whether the
  real-camera tracking system uses the same convention isn't something
  column inspection alone can settle. Flagged for Phase 3 model design,
  not blocking the loader itself (loader just needs to parse each source
  into its own SE(3) matrix; cross-source alignment is a training-time
  concern).
- **Split imbalance risk**: `_apply_split()` splits by *sequence*, and
  UnityCam is only 1 sequence out of 25 total (vs. 24 real-camera
  sequences). **Fixed 2026-08-13** — UnityCam is now sliced positionally by
  the same train/val/test fractions within its own frame range instead of
  being treated as one atomic sequence, so every split gets a share of it.
  Verified locally against a synthetic fixture (disjoint, ordered slices
  covering the full range in all three splits).

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
- 2026-08-13: Implemented `src/darkir_lite/model.py` + `train.py`. Found
  and fixed a real bug via local testing (torch is available on this
  machine): `config.yaml`'s `lr: 1e-4`/`2e-4` were parsed as *strings* by
  PyYAML, not floats (its float regex requires a decimal point in the
  mantissa) — fixed to `1.0e-4`/`2.0e-4`. Fully validated locally against
  the real cloned DarkIR repo, the real HF checkpoint, and synthetic
  frames: checkpoint loading, freeze/unfreeze gradient behavior, forward
  pass at EndoSLAM resolution and at a non-multiple-of-8 resolution,
  dataset wrapping, eval metrics, and the full train/checkpoint/resume
  cycle. Added `ptflops` to `requirements.txt` (confirmed missing on
  Kaggle's base image, required just to import DarkIR's `archs/`).
- 2026-08-13: Ran the training notebook (GPU on, `--max-steps 20` smoke
  test) — first GPU-enabled kernel in this project. Hit the P100/no-Pascal-
  kernels issue (see "GPU compatibility issue" above) twice, including a
  failed attempt to force a T4 via `--accelerator`. Added a real-op device
  check with CPU fallback instead of chasing the accelerator string
  further. First CPU run technically worked but took 40+ minutes because
  `evaluate()` ran the *full* 238-batch validation set regardless of
  `--max-steps` — fixed to cap validation at 5 batches during smoke tests.
  Re-ran: kernel `endoslam-phase2-darkir-training` v5 `COMPLETE` in ~10 min
  on CPU — train_loss=0.0007, val_psnr=23.68, val_ssim=0.772, checkpoint
  saved and reloadable. **Pipeline is smoke-tested end-to-end.** Full
  20-epoch run still needs a real GPU path (CPU is far too slow for it).
- 2026-08-13: Checked on the full 20-epoch run kicked off after the GPU fix
  (v7). `kaggle kernels status` returned `COMPLETE`; downloaded kernel
  output and found `epoch_19.pt` already present, meaning the run finished
  in a single Kaggle session — no resume cycle was needed after all.
  Loaded the checkpoint locally (`torch.load`, CPU) to confirm: `epoch=19`,
  `global_step=43240` (exactly 2162 steps/epoch x 20), `val_psnr=32.40`,
  `val_ssim=0.9226`, a clean improvement over the smoke test. Stopped the
  in-progress download of ~236 intermediate step checkpoints (~9GB, not
  needed) once the final metrics were confirmed from `epoch_19.pt` itself.
  Decided not to commit the final checkpoint into the repo — it'll be
  re-fetched from Kaggle kernel output whenever Phase 3/4 need it.
  **Phase 2 is fully done.** Next: pose parsing for Phase 3.
- 2026-08-13: Fixed the `_apply_split()` UnityCam imbalance risk (real-cam
  sequences keep the original whole-sequence shuffle; UnityCam now sliced
  positionally within its own frame range so every split gets a share),
  verified locally against a synthetic fixture. Ran the `phase3a_explore`
  kernel (v1, `COMPLETE`) to inspect real pose files for the first time —
  see "Pose format — confirmed facts" above. Implemented
  `_load_real_camera_poses()` (joins on `ImageFrame`, parsed from the xlsx
  and from `frame_NNNNNN.jpg` filenames) and `_load_unitycam_poses()`
  (positional, truncated to the shorter of frame/pose counts), normalizing
  both to `(4,4)` SE(3) matrices. Downloaded one real `.xlsx` and the real
  UnityCam `.csv` locally (`kaggle datasets download -f <path>`) to
  validate end-to-end rather than trusting the kernel log's summary stats
  alone — this caught a real bug the log didn't show: the UnityCam CSV's
  last row is a partial write (`NaN` in `rY/rZ/rW/time(s)`), which crashed
  `scipy`'s quaternion conversion until the loader added an explicit
  NaN-row drop (safe here since it's confined to the tail). Both loaders
  now validated against real data: valid SE(3) matrices (orthonormal
  rotation, correct bottom row), UnityCam's first translation matches the
  kernel log exactly (`[0.66, 8.98, -3.13]`). **Pose-parsing blocker for
  Phase 3 is resolved.** Kaggle-only validation (real 25-sequence dataset,
  full `_index_real_camera()`/`_index_unitycam()` run, `__getitem__`
  end-to-end) still pending — next step per the plan is extending
  `phase1_data_validation.ipynb` with pose/split assertion cells. Mini-3D-
  Recon model design itself (backbone/pose-head/training loop) not started.
