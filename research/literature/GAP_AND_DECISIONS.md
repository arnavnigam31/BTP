# Research gap and decision log

## 16 September: evidence-driven refinements

**D001 — Keep the main direction provisional.** Reliability-conditioned cross-task cooperation remains a candidate. We have not established that CRSDUN suffers harmful feedback after adequate training, or that a suitable solution is novel.

**D002 — Correct the loss-balancing framing.** Static loss-weight tuning is already tested in CRSDUN supplementary Table 5 (claim SUPP-01). Our earlier plan's phrase “not fully explored” must not become “not studied.” Adaptive weighting needs a comparison against a tuned static sweep, GradNorm and/or another justified established control.

**D003 — Reconcile dataset provenance before exact reproduction claims.** See DATA-01 and DATA-02. The original FVgNET description and CRSDUN's processed protocol differ in mask coverage, taxonomy and spectral representation. A different processed release or additional annotations could explain this; no explanation has yet been verified. Our public252 pipeline must retain its own identity.

**D004 — Keep uncertainty types distinct.** Hardware-mask uncertainty, global task-loss weighting, predictive confidence and reliability of cross-task feedback are different quantities. The related work must define which one we estimate and what evidence it predicts.

**D005 — Treat local fixes as infrastructure.** The transpose/progress fixes are not a new architecture. The two-crop diagnostic is not an adequate reproduction, an ablation against the published model, or evidence that a novel gate will help.

## Candidate hypotheses and falsification tests

| ID | Hypothesis | Test | Evidence against the hypothesis |
| --- | --- | --- | --- |
| H001 | Incorrect semantic feedback can damage spectral reconstruction in a trained joint model. | Perturb intermediate semantic inputs while controlling stage, feature scale, model state and measurement; measure reconstruction/segmentation changes. Compare true/estimated/random/disabled feedback with care about distribution shift. | No reproducible damage, or perturbation effects explained entirely by out-of-distribution feature magnitude. |
| H002 | Measurement consistency plus semantic stability identifies when cooperation is useful better than confidence alone. | Evaluate signals against errors and benefit of feedback on held-out validation scenes; include confident mistakes and minority classes. | No predictive value beyond a constant/stage gate; calibration alone explains results. |
| H003 | Reliability conditioning improves the joint accuracy–robustness tradeoff. | Compare under paired clean/noisy/mismatched conditions with matched training, tuning and parameter budgets. | Improvement vanishes against tuned static weights or corruption augmentation, or sacrifices one task without a justified tradeoff. |
| H004 | Stage-aware balancing addresses an observed training conflict. | Measure task-gradient magnitudes and directions on parameters reached by both losses, then compare established balancing baselines. | No conflict, no consistent gain, or benefit is only extra compute. |

## Minimum controls before a contribution claim

Aligned original method; tuned static loss weights; matched corruption augmentation; constant/stage-scheduled gate; confidence-only gate; parameter-matched unconstrained learned gate; physically informed gate; supporting loss balance alone if used; full method. Screen controls cheaply, then use repeated seeds for decisive comparisons. Ground-truth/oracle gates are diagnostic ceilings, never deployable comparisons.

Missing comparisons are not findings. Code availability does not establish reproduction feasibility. A small search returning no exact match does not establish first-of-kind novelty.

## Immediate sequence

1. Resolve public-versus-author dataset protocol and confirm university GPU model/VRAM/access.
2. Complete a balanced small-sample learning check and reproducible baseline configuration; fix explicit validation/test selection and resumable training.
3. Train an adequate aligned baseline while reading the closest methods in depth.
4. Run stage-wise failure analysis. Decide whether H001/H002 justify a minimal new module.

No additional training was launched as part of this literature-review pass.
