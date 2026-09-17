# Kaggle 1,000-step fixed-crop diagnostic

Evidence: user-supplied console.txt and submitted_prediction.png, archived here on 17 September 2026. Parsed logged evaluations: metrics_from_console.json. Reported remote output: `/kaggle/working/fixed_crop_1000_20260917_145850`.

Final step 1000: MSE 0.00034684472484514117; cross-entropy 0.2876781225204468; mIoU over the six foreground classes present in two fixed training crops 57.75827666973424%; correct-class foreground recall 77.66051862968163%. Logged learning/evaluation interval 1227.94 seconds, excluding the preceding audit. Effective AMP updates are not established by attempted-step count alone.

Best logged present-class mIoU: 77.771059581675% at step 860. At that point real-orange IoU was 84.52% and fake-pepper IoU 59.04%; at step 1000 these dropped to 44.90% and 11.55%. Thus these classes can be learned better than their final predictions suggest. At step 1000 real-grape IoU was 38.25%, fake-grape 97.58%, fake-lemon 85.23%, real-plant 69.04%. Oscillation, rather than a complete inability to learn the weak classes, is evident. Its cause is not established by these logs.

Relative to the separate 200-step run: reconstruction MSE improves from 0.00201868 to 0.000346845; final six-class mIoU improves from 50.67% to 57.76%; final cross-entropy rises from 0.26003 to 0.28768. These are training-crop diagnostics, not validation/test or paper-comparable results. Stable near-perfect multiclass memorization remains unproven.

The supplied image closely resembles the earlier 200-step prediction and appears inconsistent with the current class-count evidence (notably orange predictions). Its run identity is uncertain. Request display of predictions.png and the final metrics.json row from the exact reported 1,000-step directory before drawing new visual conclusions. No additional training is needed for that check.

Source inspection confirms the diagnostic writes final predictions from evaluate(step=1000), but saves only the final checkpoint. The step-860 model is not saved by this script. Do not claim that the best logged checkpoint is available, or choose diagnostic training-crop checkpoints for a generalization claim. A later controlled optimization diagnostic could save best and final states and log effective updates; changing the architecture or dataset is not justified by the present evidence.

Follow-up image supplied by the user confirmed final step 1000 and mIoU 0.5775827666973424. It shows restored orange/pepper detections and substantial class confusion in the plants and real-grape region. The previous image mismatch is resolved; the confirmed image remains in the local conversation archive.
