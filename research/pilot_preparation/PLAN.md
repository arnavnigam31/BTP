# Public252 baseline pilot

## Evidence and decision

The earlier paired diagnostic completed 500 Adam updates per arm. Final six-present-class training-crop mIoU was 0.7183995 at LR 4e-4 and 0.6538332 at 4e-5. These are not held-out metrics. Only comparison scores survived; comparison checkpoint/image files were not recovered from Kaggle outputs. No claim of successful near-perfect memorization is made.

Reproducing the original training-only selection gives plant scene 2021-11-03_103 at row170,col31 and mixed scene 2021-11-05_015 at row170,col55. The latter contains 1770 of its scene's 5941 class14 (fake pepper) pixels (29.79%); that class touches the crop boundary. This supports investigating coverage, not concluding that truncation caused the failure. The two crops cover only six of 22 foreground classes. Full training labels contain class14 in 31 scenes. Complete counts are in crop_coverage.json. No dataset payload was changed or test pixels inspected.

Proceed with a bounded baseline feasibility/learning pilot: fresh initialization, all 202 training scenes, 25 validation scenes each epoch, batch1, workers0, LR4e-4, lambda_seg1e-4, seed3407. Stop after epoch3, keeping the cosine scheduler horizon at500. Do not set --epochs3: that would change the learning-rate schedule. Existing crop sampling and loss remain unchanged. The failed memorization criterion remains an open diagnostic issue; this pilot tests whether it persists under broader training coverage.

## Run and retention

Use KAGGLE_BASELINE_PILOT.ipynb cells in order on a CUDA GPU. It assumes prepared public252 input and an extracted /kaggle/working/BTP updated from BTP-code-pilot.zip. Run the preflight first. The training cell always attempts to archive the run folder when it exits, including on a normal Python exception; forced session termination cannot guarantee this. Download the archive before stopping the session and check that it contains model/last.pt, both best weight files, result CSV/JSON files, protocol and provenance. Notebook cell output alone does not preserve checkpoint files. If training fails before completing epoch1, a resumable checkpoint may not exist.

Resume using the complete extracted run folder, --resume <run>/model/last.pt, --epochs500, identical batch/seed/workers and a later absolute --stop-after-epoch. Last.pt includes optimizer, scaler, scheduler and RNG state. The stop limit itself may change on resume. Transfer the complete model folder to preserve previously selected best weights.

## Review after epoch3

Inspect per-class validation IoU and recall, reconstruction PSNR_ref1/SSIM_ref1/MSE, training losses, actual optimizer updates and skipped AMP updates, runtime and peak allocated GPU memory. The validation mIoU averages all22 foreground IDs and must not be compared numerically with the earlier six-present-class crop metric as if equivalent. Three epochs do not establish convergence or journal-level improvement. If numerics/memory/checkpoint retention fail, repair that issue first. If learning is viable, extend this same schedule in bounded sessions. Persistent weak-class results motivate one controlled training-only sampling or loss experiment; do not change several factors at once or tune against test data.

## Validation

Python compilation, frozen-protocol preflight and existing CPU exact-resume/best-weight-handoff regression checks passed. The updated production training loop still requires its first Kaggle GPU pilot; no claim of a completed GPU pilot is made.
