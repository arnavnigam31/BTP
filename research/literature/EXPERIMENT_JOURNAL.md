# Experiment and project journal

Published findings are in `CLAIMS_LEDGER.md`. This file records our work; local results must never be substituted for published results.

## E001 — Earlier smoke test, 14 September 2026

Evidence: `/home/arnav/btp-cassi/src/CRSDUN/public252_smoke_e1_console.log`; inspected in this conversation. One epoch, batch size one, 202 training and 25 validation batches. Logged validation PSNR 20.68 dB, SSIM 0.5631, foreground mIoU zero; checkpoint saved.

Interpretation revised on 16 September: image and label axes were misregistered in the public candidate loader. Preserve this run as historical pipeline evidence, not a corrected baseline; do not resume aligned training from its checkpoint.

## E002 — Read-only audit and alignment inspection, 16 September

Evidence: `../../PUBLIC252_DIAGNOSTIC_NOTES.md`, `../../alignment_comparison.png`, and the WSL run's `audit.json`.

202/25/25 scene IDs, no ID overlap, expected files present. Training and validation cubes are finite and match label dimensions. All expected classes occur in both splits. Four inspected training scenes show improved registration when the image's spatial axes are transposed. This is visual evidence for those scenes, not an exhaustive registration proof. ID disjointness does not establish absence of related-scene/content leakage.

Changes: separate training batch/stage counters; explicit in-memory `--transpose_image` option across training/evaluation. Original dataset files and links unchanged. CPU checks cover crop/flip registration, option propagation, validation membership and progress timing.

## E003 — Unaligned diagnostic interrupted

Evidence: `/home/arnav/btp-cassi/runs/public252_diagnostic_20260916_console.log`.

Stopped after visual inspection revealed misalignment. No completed comparative result is claimed; this is not a controlled before/after training comparison.

## E004 — Aligned two-crop diagnostic completed, 16 September

Evidence: `../../diagnostic_summary.json`, `../../aligned_predictions.png`, diagnostic source `../../diagnostic_public252.py`; complete run under `/home/arnav/btp-cassi/runs/public252_aligned_diagnostic_20260916/`.

- Five-stage CRSDUN initialized from scratch; seed 3407; two highest-foreground distinct training scenes, fixed 256×256 crops and fixed mask; no augmentation or learning-rate decay.
- Scenes: `2021-11-03_103`, `2021-11-05_015`. Adam 0.0004; reconstruction weight 1, segmentation 0.0001, stage discount 0.7; mixed precision.
- 40 attempted updates, 39 effective Adam updates. Initial nonfinite gradients triggered one skip; scaler ended at 32768. All final model tensors finite.
- Final-stage MSE: 0.61008 → 0.02223; cross-entropy: 3.15308 → 0.83468.
- Mean IoU over six foreground classes present in these crops: 1.03% → 9.77%. Final real-plant IoU 49.79%; fake-grape 8.84%; four other present foreground classes zero.
- GPU diagnostic time approximately 822.5 seconds. This is not a server/full-training timing benchmark.

Interpretation: some foreground learning and recognizable shapes, with strong dominant-class bias. Not a passed multi-class overfit test, validation result, generalization result, or demonstrated improvement over published CRSDUN. No test pixels were used. Selection intentionally favors foreground-rich crops and is not representative sampling.

## E005 — Research scope and review

User objective: a journal paper improving on CRSDUN. Target manuscript completion: end of November 2026. University GPU access may provide about 16 GB, but GPU model, VRAM versus host memory, access time and limits remain unverified. Journal not selected. Candidate research direction remains provisional.

Git context: baseline commit `0771baf3af0c178eb9233eb7f89b7ae88480455f`, branch `btp-public252-baseline`. Alignment/progress and diagnostic changes were uncommitted at last check. GitHub push postponed at user's request. Do not associate the uncommitted fixes with the baseline commit alone.

## E006 — Full labeled-subset integrity and preprocessing audit, 16 September

Evidence: [audit report](../dataset_audit_20260916/REPORT.md), [machine-readable summary](../dataset_audit_20260916/audit.json), and [per-scene hashes](../dataset_audit_20260916/scenes.json). Read-only audit of all 252 labeled scenes, including test files for integrity/conversion verification only; no model evaluation or tuning.

Local metadata contains 315 scenes: 252 mask-labeled and 63 without masks. All six required source/derived image-label collections match the 252 labeled IDs exactly. Every checked conversion is valid; no nonfinite values or exact duplicate candidate cubes found. Interpolation maximum absolute error 1.78e-15. Every class occurs in each split. This does not establish completeness of original 317-scene FVgNET or identity with the 306-scene CRSDUN protocol.

New preprocessing concerns recorded: six starfruit scenes use our unknown-class mapping; exact author wavelength grid and upstream radiometric normalization remain unverified; training values exceed one while metrics use a reference range of one. No arbitrary scaling or clipping applied. Existing symlinks are valid but not portable without their targets. Dataset unchanged.

## Required record for every future run

Run ID/date; hypothesis and comparison; commit plus dirty diff; dataset/split/version hashes; preprocessing and mask/operator; configuration; seed; initialization; optimizer/scheduler/AMP and actual updates; hardware; logs/checkpoint paths; validation-selection rule; metric definitions and denominators; failures; result and uncertainty; decision. For corruptions, record the true acquisition operator separately from the operator supplied to the model.

## E007 — Public252-v1 preprocessing decisions, 16 September

See [resolution and primary evidence](../preprocessing_resolution/RESOLUTION.md) and [frozen specification](../preprocessing_resolution/public252-v1.json). Original Hyplex notebooks support transpose and unknown-class fallback; active loader uses no extra normalization or clipping. Retained intensities and unit-reference metrics, fixed existing candidate wavelengths and explicit starfruit mapping. Read-only preflight passed all 252 scene headers, metadata hashes, one training-scene loader check and synthetic metrics. No dataset changes or training. Exact CRSDUN processed-release equivalence remains unverified: Baidu browser access blocked by site-safety policy.


## E008 — Portable training handoff, 16 September 2026

CPU resume/split/augmentation checks passed. One real-crop CUDA optimizer update and checkpoint reload passed; [result](../preprocessing_resolution/local_cuda_smoke.json). No baseline or held-out test evaluation was run. Single-GPU epoch-boundary resume, separate best-IoU/best-PSNR weights, explicit eval splits and CSV metrics implemented. Multiclass overfitting and full-resolution validation memory tests remain pending. See [server instructions](../../SERVER_SETUP.md).

## E009 — Testing readiness, 17 September 2026

Production evaluation tested on deterministic CPU fixtures: expected MSE/PSNR, confusion matrix and scene IDs; invalid outputs rejected. Plain, DDP-prefixed and full checkpoint weights load correctly. Precision retained before metric aggregation and best-checkpoint selection. Default evaluation outputs metrics only; full predictions opt-in. Dataset remains public252-v1; checkpoint selection behavior changed from rounded legacy scores and must be recorded. No held-out model evaluation or Kaggle execution performed. Local test results: [checks](../preprocessing_resolution/local_cpu_checks.txt).

Transfer-integrity verification on 17 September: all 252 candidate cube array hashes, all 252 mapped label file hashes, and label shapes/palettes passed. Resume into a different output folder preserves prior best-IoU/best-PSNR weights and rejects conflicting files; regression passed. Dataset files unchanged.
