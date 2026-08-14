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
| 1 | 2-5 | Dataset loader working, synthetic dark-degradation validated | **Done and validated on Kaggle** (kernel `endoslam-phase1-validation`, version 8, `COMPLETE`, 2026-08-13). `_index_sequences()` rewritten to match the real layout (see below); pose parsing and the split-imbalance fix (originally deferred) are now also implemented and validated — see Phase 3 prep notes below. `train_ds`/`val_ds`/`test_ds`: 16836/2032/2736 windows (post pose/split fix; was 17150/1885/2588 pre-fix). Dark-degradation visual check ran without error. |
| 2 | 5-10 | DarkIR-lite fine-tuned from pretrained checkpoint, PSNR/SSIM logged | **Done.** Full 20-epoch fine-tune completed in a single Kaggle session on GPU (kernel `endoslam-phase2-darkir-training`, `COMPLETE`, 2026-08-13) — no resume cycle needed. Final checkpoint `epoch_19.pt`: `global_step=43240` (= 2162 steps/epoch x 20, exact), **val_psnr=32.40, val_ssim=0.9226** — up from the 20-step smoke test's 23.68/0.772. Checkpoint left in Kaggle kernel output only (not committed to the repo); re-fetch via `kaggle kernels output ritiksharma8/endoslam-phase2-darkir-training` when Phase 3/4 need the trained weights. |
| 3 | 10-20 | Mini-3D-Recon trained on UnityCam depth+pose GT | **Done.** `MiniReconModel` (MobileNetV3-Small backbone + depth/pose heads, ~3M params) trained the full 40 epochs in one Kaggle session (kernel `endoslam-phase3-mini3drecon-training`, `COMPLETE`, 2026-08-13) — no resume needed. Final checkpoint `epoch_39.pt`: `global_step=12280` (= 307 steps/epoch x 40, exact), **val_depth_absrel=0.118, val_rot_err_deg=0.46°, val_trans_err=0.00357** (raw UnityCam world-units) — all improved sharply from the smoke test's 0.93/14.50°/0.031. Checkpoint left in Kaggle kernel output only (not committed); re-fetch via `kaggle kernels output ritiksharma8/endoslam-phase3-mini3drecon-training` when Phase 4 needs it. |
| 4 | 20-25 | Pose chaining + depth backprojection -> Open3D point cloud viewer | **Done.** Kernel `endoslam-phase4-reconstruction`, `COMPLETE` — GT-mode reconstruction produced a coherent stomach-lumen tube shape (not scattered), passing the empirical gate for the sourced-but-unconfirmed camera model. See "Phase 4 reconstruction run" below. |
| 5 | 25-30 | With/without-DarkIR comparison, ATE/RPE + AbsRel/RMSE, report | **Done.** Kernel `endoslam-phase5-evaluation` v2, `COMPLETE` — `darkir_lite_enhanced` beat `raw_dark_input` on every metric (AbsRel -30%, RMSE -34%, delta1 +71% relative, ATE -20%, RPE trans -38%, RPE rot -48%). See `REPORT.md` and "Phase 5 evaluation run" below. |

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

## Depth format — confirmed facts (2026-08-13, `phase3b_depth_explore` kernel v1)

`_index_unitycam()` globbed `Pixelwise Depths/*` with no extension filter
and read it via `cv2.imread(..., IMREAD_UNCHANGED)` — nobody had ever
checked this data's format. Real facts, from `kaggle kernels output
ritiksharma8/endoslam-phase3b-depth-explore`:

- **File format**: `.png`, 8-bit, **4-channel (RGBA)** — e.g.
  `aov_image_0000.png` ("AOV" = Arbitrary Output Variable, a render-engine
  term for an auxiliary render pass). All four channels are identical
  (sample row: `[10 10 10 10]` repeated) — a single scalar replicated for
  display, not a multi-channel precision-packing scheme.
- **Value range**: observed `min=1, max=72` across 6 frames sampled across
  the whole sequence (not spanning the full 0-255 range) — clearly not
  meters (would need mm-level precision at this scale) and not obviously
  any other standard unit either.
- **Absolute units are NOT confirmed.** Searched the official EndoSLAM
  repo (`github.com/CapsuleEndoscope/EndoSLAM`) and its README/eval code —
  no documented conversion formula for the Unity depth export found. A
  "raw millimeters" hypothesis is physically plausible (stomach endoscopy
  working distances are typically single-digit to low-double-digit mm,
  matching the observed 1-72 range almost exactly) and consistent with
  `config.yaml`'s `fusion.depth_trunc: 0.15` (150mm), but this is an
  inference, not a confirmed fact.
- **This does not block Phase 3 training**, because the official repo's
  own `EndoSfMLearner/eval_depth.py` applies **median-ratio scaling**
  between predicted and GT depth (`ratio = median(gt)/median(pred)`)
  before computing AbsRel/RMSE/delta1 — i.e. the reference methodology
  itself treats predicted-vs-GT depth as scale-ambiguous rather than
  assuming a known absolute unit. Phase 3 trains directly against the raw
  pixel values (as float32) with a plain masked loss; true-metric
  calibration, if ever needed, is a Phase 4 point-cloud-fusion concern,
  not a Phase 3 training-time one.
- **`has_depth` reconfirmed ~100%** for UnityCam: frame count (1548) ==
  depth count (1548) exactly, matching the earlier dataset-loader
  validation.
- **Real bug found and fixed** (`src/data/endoslam_dataset.py`
  `__getitem__`): `cv2.imread(path, IMREAD_UNCHANGED)` on a 4-channel PNG
  returns `(H,W,4)`; `cv2.resize` preserves the channel count, so the
  previous code silently produced `depths` of shape `(T,H,W,4)` instead
  of the documented `(T,H,W)` — never caught because prior validation only
  asserted pose tensor shapes, not depth shapes. Fixed by taking channel 0
  (`d[..., 0]`) right after `imread`, before resize, since all four
  channels are confirmed identical.

## Phase 4 camera model — sourced but UNCONFIRMED (2026-08-13)

Depth backprojection to a 3D point cloud needs camera intrinsics
(focal length) and a way to convert UnityCam's raw depth-byte values into
real scene-scale distance — neither has ever existed anywhere in this
project, the EndoSLAM paper, or its repo (searched exhaustively; the
paper's Table C.2 gives real intrinsics for HighCam/LowCam/MiroCam/PillCam
via checkerboard calibration, but UnityCam is a synthetic Unity camera,
never calibrated, and is absent from that table entirely).

Found a real, sourced candidate by going one level further: the EndoSLAM
README points to `github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy` for
"generation of synthetic data." Its
`VR-Caps-Unity/Assets/Scenes/Record_scene.unity` (plain-text Unity scene
YAML) contains a `Camera` GameObject (fileID 579270852) positioned near a
physics-simulated `Capsule` GameObject (fileID 551316796, has a
`Rigidbody` with `m_Mass: 0.01`), with:
```
m_FOVAxisMode: 0   # 0 = Vertical
field of view: 91.320755
near clip plane: 0.01
far clip plane: 2
```
Two other cameras exist in the same scene — a "Main Camera" with untouched
Unity defaults (FOV 67.38°, sensor 36x24mm — clearly not endoscope-
specific) and a "MeshGenerationCam" (FOV 77°, used for the separate
`3D_Scanners` mesh exports, not per-frame video) — the `Camera` object is
the best candidate based on its close-range near/far clip (physically
plausible for a capsule endoscope inside a stomach) and proximity to the
physics-simulated capsule.

**This is NOT 100% confirmed**: can't verify this exact scene generated
the specific stomach dataset variant mirrored on Kaggle (`mcocoz/
endoslam`), and the repo's other scene file (`Clinic Setup.unity`) wasn't
checked. Also unconfirmed: the depth-byte-to-distance conversion
(`near + byte/255*(far-near)`, the "Linear01Depth" convention — plausible
since the project uses Unity's HDRP render pipeline, confirmed via an
`HDRPDefaultResources` folder, whose depth AOV is conventionally output
this way, but not verified for this specific export) and the
backprojection axis convention (Unity is left-handed y-up; standard CV
pinhole backprojection assumes a different convention — Phase 3's
rotation-matrix loss never needed to care about this, so it's untested).

**Both unconfirmed items get an empirical visual gate before being
trusted**: `src/fusion/reconstruct.py::reconstruct_gt()` backprojects the
dataset's own GT depth+pose (no model involved) — if it produces a
coherent stomach-lumen tube shape rather than a scattered mess, that's
real evidence the hypothesis holds; if not, revisit before trusting any
model-based (`reconstruct_predicted()`) output. See the Log below for the
outcome once run.

## Immediate next steps

1. Phase 3: implement `_load_poses()` against the confirmed formats above
   — real-camera loader joins on `ImageFrame` (parsed from both the xlsx
   column and the `frame_NNNNNN.jpg` filename), UnityCam loader truncates
   positionally to `min(frame_count, pose_row_count)` since there's no
   alignment key. **Done** — see the Log entry below.
2. Phase 4: run `phase4_reconstruction` on Kaggle, visually validate the
   GT-mode reconstruction (resolves the camera-model hypothesis above),
   then trust/inspect the predicted-mode output. **Done** — see "Phase 4
   reconstruction run" below.
3. Phase 5: implement `src/eval/metrics.py` (depth AbsRel/RMSE/delta1 with
   median-ratio scaling, ATE/RPE via Umeyama-aligned trajectories) and
   `src/eval/run_comparison.py` (raw-dark vs DarkIR-enhanced input through
   the trained Mini-3D-Recon model on the UnityCam test split, fixed-seed
   degradation so both conditions see identical dark input), run on Kaggle,
   then write the final `REPORT.md`. **Done** — see "Phase 5 evaluation run"
   below and `REPORT.md`.

All 6 phases (0-5) are now done. The project's core hypothesis (brightening
first improves reconstruction) is confirmed with numbers — see `REPORT.md`.
Remaining work, if any, is polish/extension, not a required next phase.

## Phase 5 evaluation run (2026-08-14, `endoslam-phase5-evaluation` kernel)

Ran `phase5_evaluation.ipynb` on Kaggle (GPU off, inference-only, UnityCam
test split = 148 windows).

- **v1 failed**: `ValueError: too many values to unpack (expected 4)` inside
  `DarkIR.forward()`'s `_, _, H, W = input.shape`. Root cause:
  `run_comparison.py` called `darkir_model(dark_images.unsqueeze(0)...)`,
  copying `MiniReconModel`'s `(B,T,3,H,W)` temporal-window calling
  convention onto DarkIR, which has no notion of a temporal window and
  expects plain `(B,3,H,W)` (same as `darkir_lite/train.py` already calls
  it) — the extra `unsqueeze(0)` produced an invalid 5D tensor. Fixed by
  passing the window's `T` frames as DarkIR's batch dimension directly. The
  local smoke test hadn't caught this because its DarkIR stand-in
  (`nn.Identity()`) doesn't enforce a shape contract — strengthened the
  smoke test to assert 4D input, matching DarkIR's real constraint, so this
  class of bug is now caught locally next time.
- **v2 (`COMPLETE`)**: every metric improved with `darkir_lite_enhanced`
  over `raw_dark_input` — depth AbsRel 0.327->0.227 (-30.4%), RMSE
  11.011->7.244 (-34.2%), delta1 0.398->0.680 (+71% relative), ATE
  0.00724->0.00582 (-19.6%), RPE translation RMSE 0.001546->0.000963
  (-37.7%), RPE rotation RMSE 1.127°->0.582° (-48.4%). Preview triplets
  (dark | DarkIR-enhanced | clean GT) show the enhanced frame visually close
  to the clean reference in all 3 sampled examples. Full numbers in
  `phase5_metrics.json` (not committed — regenerate via the notebook);
  preview images copied into `report/previews/` and committed alongside
  `REPORT.md`.
- Also fixed a real bug in `src/data/dark_degradation.py` found via local
  testing before this run: `random_motion_blur_kernel()` used the global
  `np.random` state for its blur angle instead of the `rng` parameter
  `degrade_frame()` already threaded through everywhere else — meant
  `degrade_frame(..., rng=seeded_rng)` wasn't actually fully deterministic,
  which would have broken Phase 5's "both conditions see identical dark
  input" fairness requirement silently.

## Phase 4 reconstruction run (2026-08-13, `endoslam-phase4-reconstruction` kernel)

Ran `phase4_reconstruction.ipynb` on Kaggle (GPU off, inference-only —
~3M-param model, ~1541 forward passes, CPU-tractable): `COMPLETE`. Output
downloaded (`kaggle kernels output`): `pointclouds/{gt_ydown_True,
gt_ydown_False,predicted}.ply` + matching `previews/*.png` (matplotlib
scatter, 3 orthographic views each — headless-safe, no OpenGL dependency).

- **GT-mode sweep (`y_down` True vs False)**: both produced a coherent,
  continuous, branching stomach-lumen tube shape — neither was a scattered
  mess. This passes the empirical gate `reconstruct_gt()`'s docstring and
  the notebook's own "Done" cell describe: the sourced-but-unconfirmed
  camera model (FOV 91.32°, near/far clip 0.01/2.0, Linear01Depth
  byte→distance conversion — see "Phase 4 camera model" above) is now
  **empirically supported**, not just sourced.
- **`y_down` decision**: the sweep wasn't a stark differentiator (both
  looked like plausible tubes), so kept the existing `configs/config.yaml`
  default (`depth_axis_y_down: true`) rather than flipping it on weak
  evidence — this is also the setting `predicted.png` was generated under.
- **Predicted-mode**: run under `y_down=true`, anchored at GT frame 0's
  absolute pose (`anchor_predicted_to_gt_origin` — world-frame origin
  choice only, not supervision leakage). Its point cloud is structurally
  consistent with the GT-mode tube shape (similar extent/envelope, same
  branching character) — a real signal that Phase 3's model produces
  geometrically sound depth+pose, not just low loss numbers.
- **"Point cloud viewer" deliverable**: satisfied by the `.ply` files
  (openable in any Open3D-based tool locally) + the preview PNGs — no
  interactive viewer script written, since Kaggle is headless (no
  OpenGL/display) and an interactive session only makes sense locally.
  `.ply` files were not committed to the repo (same "re-fetch from Kaggle
  kernel output when needed" pattern as the Phase 2/3 checkpoints).

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
  `phase1_data_validation.ipynb` with pose/split assertion cells.
- 2026-08-13: Extended `phase1_data_validation.ipynb` with pose/split
  assertion cells and re-ran it on Kaggle (kernel
  `endoslam-phase1-validation` v8, `COMPLETE`) against the real 25-sequence
  dataset — confirms everything above at full scale, not just the 3 sampled
  trajectories/synthetic fixtures: all 24 real-camera trajectories parsed
  with pose rows (no crashes), UnityCam consistently drops 1/1544 NaN row
  and truncates to 1543 (matches the local finding exactly), and the split
  fix gives every split non-zero UnityCam windows for the first time
  (train=1227, val=147, test=148, out of 16836/2032/2736 total windows —
  window counts differ from the pre-fix run's 17150/1885/2588 as expected,
  since UnityCam's split behavior changed). Pose tensors are `(T,4,4)`, no
  NaNs, valid SE(3) bottom row, checked on all three splits. **Both Phase 3
  blockers (pose parsing, split imbalance) are fully resolved and verified
  on the real dataset.** Mini-3D-Recon model design itself
  (backbone/pose-head/training loop) not started — next planning session.
- 2026-08-13: Started Phase 3 model work. Planning decisions: train on
  UnityCam only (matches README's literal wording, sidesteps the
  unresolved real-cam/UnityCam pose coordinate-frame mismatch), pose head
  predicts consecutive frame-to-frame relative pose (matches Phase 4's
  "pose chaining" wording). Added `src/reconstruction/geometry.py`
  (6D-rotation -> SE(3), relative-pose-from-absolute), validated locally.
  Extracted `_select_device()`'s P100 workaround out of
  `darkir_lite/train.py` into `src/common/device.py` for reuse. Ran the
  `phase3b_depth_explore` kernel (v1, `COMPLETE`) to check UnityCam's
  never-before-inspected depth format — see "Depth format — confirmed
  facts" above; found and fixed a real shape bug in
  `endoslam_dataset.py`'s depth loading (`(H,W,4)` instead of `(H,W)`)
  along the way. Implemented `src/reconstruction/model.py`
  (`MiniReconModel`: MobileNetV3-Small backbone + `DepthHead` + `PoseHead`,
  ~3M params), validated locally with real pretrained weights (correct
  output shapes, orthonormal rotations, non-negative depth). Implemented
  `loss.py` (masked depth L1, gradient-isolation checked; pose loss with a
  trace-based rotation term) and `train.py` (mirrors `darkir_lite/train.py`
  conventions exactly), validated locally against a fake in-memory dataset
  (loss decreases over steps, checkpoint save/load round-trips with
  bit-identical resumed output).
- 2026-08-13: Pushed `phase3_training` to Kaggle. Smoke test (v1,
  `--max-steps 20`, `COMPLETE`): `GPU FIX CONFIRMED` (Tesla P100, torch
  2.5.1+cu121, same pin as Phase 2), `train batches: 307, val batches: 37`
  (matches the confirmed 1227/147 UnityCam windows at `batch_size: 4`), no
  NaN/crash, `rot_err_deg` dropped 37.58° → 14.50° within 20 steps (random
  init typically starts ~90-120° for 3D rotations, so real learning is
  happening despite `depth_loss` (10.57) dominating the raw loss sum over
  `trans_loss`/`rot_loss` (0.058/0.326) — judged sufficient to not block
  the real run; `pose_rotation_weight: 10.0` left as-is). Checkpoint
  round-trip confirmed on real Kaggle infra. Flipped `MAX_STEPS` to `None`
  and re-pushed for the real 40-epoch run.
- 2026-08-13: Full 40-epoch Mini-3D-Recon run (kernel v2) completed in a
  single Kaggle session — no resume cycle needed, mirroring Phase 2's
  experience. Final checkpoint `epoch_39.pt`: `global_step=12280` (exactly
  307 batches/epoch x 40), `val_depth_absrel=0.118` (down from 0.93 at the
  20-step smoke test), `val_rot_err_deg=0.46°` (down from 14.50°),
  `val_trans_err=0.00357` raw units (down from 0.031) — the model learned
  both depth and relative pose well within this dataset's own scale.
  **Phase 3 training is fully done.** `pose_rotation_weight: 10.0`'s guess
  turned out fine in practice — rotation error converged to sub-degree
  accuracy without needing adjustment. Not yet done: Phase 4 (pose
  chaining + depth backprojection -> Open3D point cloud viewer) and Phase
  5 (evaluation/report) — next planning session.
- 2026-08-13: Ran `phase4_reconstruction` on Kaggle (kernel
  `endoslam-phase4-reconstruction`, `COMPLETE`, GPU off). See "Phase 4
  reconstruction run" above for full detail — summary: both `y_down`
  sweep variants produced coherent GT-mode tube shapes, empirically
  confirming the sourced camera-model hypothesis; kept the existing
  `depth_axis_y_down: true` default; predicted-mode output (same setting)
  matched the GT tube's shape, a real Phase 3 quality signal. `.ply` +
  preview PNGs satisfy the "point cloud viewer" deliverable given headless
  Kaggle. **Phase 4 is done.** Next: Phase 5 (with/without-DarkIR
  comparison, ATE/RPE + AbsRel/RMSE/delta1, report).
- 2026-08-14: Implemented and ran Phase 5. `src/eval/metrics.py`
  (median-ratio depth AbsRel/RMSE/delta1, Umeyama-aligned ATE/RPE) and
  `src/eval/run_comparison.py` validated locally first (synthetic fixtures
  + a fake in-memory dataset) before spending Kaggle quota — this caught
  two real bugs: `dark_degradation.random_motion_blur_kernel()` ignoring
  its `rng` parameter (fixed, see "Phase 5 evaluation run" above), and
  `trajectory_metrics()`'s original RPE not correcting for the scale offset
  between predicted (raw uncalibrated units) and GT trajectories (fixed by
  scale-correcting translations before the relative-pose differencing).
  First Kaggle run (v1) still failed — a third bug local testing didn't
  catch: `run_comparison.py` called DarkIR with `MiniReconModel`'s temporal
  batching convention instead of DarkIR's own per-frame one, fixed and
  smoke test strengthened to catch this class of bug locally next time.
  v2 `COMPLETE`: **`darkir_lite_enhanced` beat `raw_dark_input` on every
  metric** (AbsRel -30.4%, RMSE -34.2%, delta1 +71% relative, ATE -19.6%,
  RPE translation -37.7%, RPE rotation -48.4%) on the 148-window UnityCam
  test split. Wrote `REPORT.md` with the full metrics table and 3 preview
  triplets (`report/previews/`). **Phase 5 is done — all 6 phases (0-5) of
  this project are now complete.** The project's core hypothesis
  (brightening first improves reconstruction) is confirmed with real
  numbers on held-out data.
