# Dark Endoscope Video -> 3D Model

Input: dark, low-quality endoscope video.
Output: a rotatable 3D point cloud of the esophagus/stomach interior, plus a
report proving (with numbers) that brightening the video first improves the
3D result.

Pipeline: DarkIR-lite (frame enhancement) -> Mini-3D-Recon (pose + depth per
frame, inspired by LingBot-Map's anchor / local-window / trajectory-memory
idea, drastically shrunk) -> point cloud fusion (Open3D) -> evaluation.

Constraints this repo is designed around: 1 person, 30 days, no local GPU.
All training happens on free Kaggle/Colab GPU time (T4-class, session limits
~9-12h, weekly quota capped). Checkpointing every N steps is not optional.

## Dataset

EndoSLAM (Ozyoruk et al., 2021) - stomach subset only.
- UnityCam (synthetic stomach, 320x320): the ONLY part with paired GT depth
  AND pose. This is the quantitative backbone.
- HighCam / LowCam (ex-vivo pig stomach, real footage, pose GT only): the
  real-video demo and credibility check.
Mirrored on Kaggle: kaggle.com/datasets/mcocoz/endoslam (verify completeness
against the official GitHub repo / Mendeley listing once you're in a
notebook -- Kaggle mirrors of research datasets sometimes lag or drop files).

## Phase plan (30 days)

| Phase | Days  | Deliverable |
|-------|-------|-------------|
| 0     | 1     | Repo, env, Kaggle/Colab pipeline, dataset confirmed accessible |
| 1     | 2-5   | Dataset loader working, synthetic dark-degradation validated |
| 2     | 5-10  | DarkIR-lite fine-tuned from pretrained checkpoint, PSNR/SSIM logged |
| 3     | 10-20 | Mini-3D-Recon trained on UnityCam depth+pose GT (highest-risk phase) |
| 4     | 20-25 | Pose chaining + depth backprojection -> Open3D point cloud viewer |
| 5     | 25-30 | With/without-DarkIR comparison, ATE/RPE + AbsRel/RMSE, report |

If Phase 3 runs long, cut here first: shorter context window (4 frames
instead of 8), smaller backbone, fewer epochs -- not the eval or the report.

## Status

See `PROGRESS.md` for current phase status, open TODOs, and next steps.
