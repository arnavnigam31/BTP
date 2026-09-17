# Kaggle fixed-crop diagnostic — 17 September 2026

Evidence: user-supplied [console output](console.txt). Run directory reported by Kaggle: `/kaggle/working/fixed_crop_20260917_144904`. Earlier environment report: Tesla T4, torch 2.10.0+cu128. This analysis did not access the remote checkpoint or prediction image.

Two fixed training crops, six present foreground classes, 200 attempted steps. The console does not establish the number of effective AMP optimizer updates. No test pixels read by this diagnostic; no validation model metrics reported. Approximately 252.14 seconds for the logged learning/evaluation interval, excluding the preceding dataset audit.

| Metric | Initial | Step 200 |
| --- | ---: | ---: |
| Reconstruction MSE | 0.61007747 | 0.00201868 |
| Segmentation cross-entropy | 3.15307486 | 0.26002839 |
| mIoU over six present foreground classes | 1.02825% | 50.67143% |
| Correct-class foreground recall | 4.20519% | 82.12551% |

Final class IoUs: real orange 0.56090%, real grape 71.60595%, fake grape 90.51909%, fake lemon 55.50653%, fake pepper 2.76836%, real plant 83.06775%. Other foreground IDs are absent from the selected crop targets; their zero entries do not establish failure to learn those classes across the full dataset.

The highest logged present-class mIoU was 59.44629% at step 190; it fell to 50.67143% at step 200 even though cross-entropy and reconstruction MSE decreased. Only the final diagnostic checkpoint is saved by the current script; do not label it a step-190 checkpoint.

Decision: infrastructure and substantial multiclass learning demonstrated; near-perfect fixed-crop overfitting not demonstrated. Two present classes remain poorly learned and class predictions fluctuate. This result alone does not diagnose a loader bug, establish generalization, or justify changing the architecture/loss. Inspect `predictions.png` against labels next, then consider a longer unchanged-settings diagnostic. Full-frame validation GPU memory remains untested. Do not compare this six-class training-crop metric with the paper's foreground benchmark score.
