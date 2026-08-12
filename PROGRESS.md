# Progress

Canonical status doc. `README.md` has the project pitch and phase plan;
this file tracks what's actually done. Update this, not the README
checklist, as work progresses.

## Where code runs

- **Local (this machine)**: all authoring — source files, config, the
  notebook itself. No GPU, no dataset here, so nothing here actually
  executes against real data.
- **Kaggle**: GPU execution, dataset access (`/kaggle/input/endoslam`),
  training, checkpoints, evaluation. Primary platform — matches
  `config.yaml`'s data root and the README's Kaggle dataset mirror link.
  Colab is a fallback only if Kaggle's weekly GPU quota becomes a
  blocker; no Colab-specific code exists yet.
- **Round-trip rule**: code reaches Kaggle via `git clone` inside the
  notebook (GitHub is the single source of truth for `src/`), not
  manual upload or copy-paste. Any fix discovered while running on
  Kaggle — e.g. patching `_index_sequences()` once the real folder
  layout is known — must be copied back into local `src/` and committed
  here, not left stranded in a Kaggle notebook edit.
- **GitHub remote**: not yet set up (deliberate — deferred until the
  user is ready to run on Kaggle, since `git clone` from a notebook
  needs a pushable remote). Add one before running
  `phase1_data_validation.ipynb` for real.

## Phase status

| Phase | Days  | Deliverable | Status |
|-------|-------|-------------|--------|
| 0 | 1 | Repo, env, Kaggle/Colab pipeline, dataset confirmed accessible | Repo scaffold + `config.yaml` + `requirements.txt` done and committed. Dataset accessibility **not yet confirmed** — needs the `os.walk` inspection on Kaggle. |
| 1 | 2-5 | Dataset loader working, synthetic dark-degradation validated | `EndoSLAMStomachDataset` and `dark_degradation.py` written, **untested against real data**. `_index_sequences()` is a placeholder guess at folder structure — see TODO below. Notebook to validate this is next. |
| 2 | 5-10 | DarkIR-lite fine-tuned from pretrained checkpoint, PSNR/SSIM logged | Not started. |
| 3 | 10-20 | Mini-3D-Recon trained on UnityCam depth+pose GT | Not started. Highest-risk phase per README — cut context window/backbone/epochs first if time runs short, not eval/report. |
| 4 | 20-25 | Pose chaining + depth backprojection -> Open3D point cloud viewer | Not started. |
| 5 | 25-30 | With/without-DarkIR comparison, ATE/RPE + AbsRel/RMSE, report | Not started. |

## Immediate next steps

1. Write `notebooks/phase1_data_validation.ipynb`:
   - `os.walk("/kaggle/input/endoslam")` inspection cell (see docstring
     in `src/data/endoslam_dataset.py`) to find the real folder layout.
   - Patch `_index_sequences()` (and `_load_poses()`'s column-count
     assumption) to match, then copy the patch back to local `src/` and
     commit.
   - Dataset-loader smoke test: sequence/window counts, batch shapes,
     confirm `has_depth` is only `True` for UnityCam.
   - Dark-degradation before/after visual check on real frames.
2. Before running that notebook on Kaggle: add a GitHub remote and push
   (currently deferred — see above).
3. Once Phase 1 is validated end-to-end, start Phase 2 (DarkIR-lite
   fine-tuning).

## Known open TODOs in code (not yet resolved)

- `endoslam_dataset.py::_index_sequences()` — folder-structure guess,
  unverified against real data.
- `endoslam_dataset.py::_load_poses()` — assumes whitespace-separated
  6-or-7-value pose rows; convention (camera-to-world vs world-to-camera)
  unconfirmed.
- `FrameSample.pose` dtype/shape ambiguity (`(4,4)` vs `(6,)`) — will
  resolve once `_load_poses()` is confirmed.

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
