# BTP research plan: journal work building on CRSDUN

Prepared 16 September 2026. This is a research plan, not a claim of established novelty or guaranteed publication.

Working evidence notebook: [literature review and claims ledger](literature/README.md).

**Review update, 16 September:** CRSDUN supplementary Table 5 already tests four static segmentation-loss weights. The research gap concerns better strategies and evidence, not an absence of loss-weight experiments. The original FVgNET paper reports masks for 80% of its 317 images; this must be reconciled with CRSDUN's processed-data protocol. See the notebook's SUPP-01 and DATA-01–03 entries. No explanation for the release difference is yet verified.

## Project objective

Develop and rigorously evaluate an original improvement to joint hyperspectral reconstruction and semantic segmentation from CASSI measurements, using He et al., CVPR 2026, as the principal baseline. The intended output is a journal manuscript with a clear methodological contribution, controlled experiments, and reproducible implementation.

The current debugging, dataset adaptation, and training scripts are supporting infrastructure. They must not be presented as the main scientific contribution.

## What the paper establishes

The paper combines CASSI measurement fidelity, a spectral-dictionary coupling between reconstructed spectra and segmentation variables, and learned joint priors. HQS-inspired unfolding alternates reconstruction and segmentation updates. CASTA exchanges information between their feature representations. Stage-specific degradation-aware estimators are already part of the method (Eq. 23).

Its loss uses multi-stage reconstruction MSE and segmentation cross-entropy, with stage weight 0.7 and segmentation weight 0.0001 (Eq. 29). Section 6 explicitly identifies task-loss balancing and segmentation robustness to noise/mask errors as unresolved issues. Section 5.4 already uses shot-noise augmentation for real-data experiments; adding noise training alone is therefore insufficient differentiation.

Table 4 reports reconstruction PSNR increasing with depth while segmentation quality is best at five stages among the tested depths. This motivates investigating stage-dependent cooperation, but does not by itself prove harmful feedback or gradient conflict.

The paper uses 256 training and 50 testing scenes after excluding 11 of 317 scenes. Our current 202/25/25 public252 split and candidate spectral preprocessing are different. Published scores cannot serve as direct matched-protocol comparisons with our development results.

Source: [CRSDUN paper](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf), especially Eqs. 8–29, Sections 5–6, and Tables 1–4. [Official code and dataset links](https://github.com/zjhe02/CRSDUN).

## Recommended candidate research question

**When should reconstruction and segmentation trust each other's intermediate predictions, particularly when CASSI measurements or calibration are imperfect?**

Working hypothesis: bidirectional cooperation can propagate unreliable intermediate estimates; conditioning the strength and direction of cooperation on measurable reliability may improve the joint reconstruction–segmentation tradeoff under measurement mismatch, while preserving clean-data performance.

This hypothesis must first be tested on an adequately trained, aligned baseline. Our short diagnostic's dominant-class predictions are not evidence that the published architecture has this failure mode.

### Candidate method

Investigate reliability-conditioned cross-task coupling within the existing unfolding framework, rather than replacing the entire backbone.

- Start with a minimal stage-specific gate controlling semantic feedback into reconstruction; then test whether reciprocal gating is warranted.
- Candidate signals include normalized measurement residuals backprojected to the image domain, prediction stability between stages, and calibrated segmentation uncertainty.
- Measurement residuals alone are not proof of reconstruction correctness because the inverse problem has a null space. Entropy alone is not proof of correctness because predictions may be confidently wrong. Test complementary signals and explicit wrong-but-confident cases.
- Keep signals available at inference; ground truth may supervise training or support oracle ablations, but must not be supplied to the deployed gate.
- If the spectral-dictionary coupling term is modified, derive the resulting data update consistently. Do not retain the original closed-form update while claiming it solves a different weighted objective. An approximate update must be labeled and evaluated as such.
- Treat stage-aware task-loss balancing as a possible supporting mechanism, not an obligatory second module. Include it only if measured gradient conflict or imbalance supports the decision.

Provisional research description: **Reliable cooperative unfolding for joint spectral reconstruction and segmentation under measurement mismatch.** Final title and novelty statement follow evidence and literature review.

### Novelty checks already identified

These are starting points, not an exhaustive review:

| Existing work | Why it matters |
| --- | --- |
| [Modeling Mask Uncertainty in Hyperspectral Image Reconstruction, ECCV 2022](https://arxiv.org/abs/2112.15362) | Mask uncertainty and hardware miscalibration are established topics. |
| [DAUHST](https://arxiv.org/abs/2205.10102) | Degradation-aware unfolding is already established. |
| [RDLUF, CVPR 2023](https://arxiv.org/abs/2211.06891) | Learning residual degradation to address operator mismatch is established. |
| [GradNorm](https://arxiv.org/abs/1711.02257) | Adaptive task weighting needs comparison against established balancing methods. |
| [PCGrad](https://arxiv.org/abs/2001.06782) | Gradient-conflict handling is not new by itself. |
| [VL-DUN, 2026 preprint](https://arxiv.org/abs/2601.23103) | Joint restoration–segmentation unfolding also exists in adjacent domains. Read in detail before framing novelty. |

Build a literature matrix including the main paper's references, subsequent citing papers, joint inverse-imaging/segmentation methods, uncertainty-conditioned feature fusion, robust CASSI, and adaptive unfolding. Record the exact difference from the closest methods. A generic gate, familiar loss, or backbone substitution is not assumed sufficient for a journal contribution.

## Work phases and decision gates

| Phase | Work | Required evidence before advancing |
| --- | --- | --- |
| 1. Reproduction and provenance | Finish balanced small-sample learning check; resolve original data/preprocessing availability; verify spectral calibration, normalization, labels, related-scene leakage, mask/shift conventions and metric definitions; add explicit evaluation split and resumable training. | Documented protocol and a stable aligned baseline. Original-paper reproduction and public252 development results clearly separated. |
| 2. Baseline training | Train three-stage and five-stage CRSDUN under recorded budgets, using validation only for selection. Preserve the paper loss as a control. | Learning curves, per-class results, stage-wise outputs, reproducible checkpoints and compute measurements. |
| 3. Failure analysis | Measure gradient norms/conflicts on genuinely shared parameters, stage-wise task quality, confidence versus error, and the effect of controlled semantic-feedback perturbations. Stress test noise and operator mismatch. | Reproducible failure tied to a plausible mechanism, beyond training or preprocessing problems. |
| 4. Minimal intervention | Implement the smallest reliability-conditioned modification and compare with fixed, constant-gate, stage-scheduled, entropy-only and parameter-matched learned-gate controls. | Gains over matched controls across seeds; mechanism matches the claimed behavior. If generic controls explain the gain, revise or drop the novelty claim. |
| 5. Full evaluation | Add relevant joint and two-stage baselines, robust-training controls, a second suitable dataset/domain if available, and real measurements if the intended claim requires them. | Accuracy, robustness and compute tradeoffs are supported under identical protocols. |
| 6. Manuscript | Write method derivation, protocol, ablations, limitations and reproducibility package alongside experiments. Select journal with supervisor once contribution and evidence are clear. | A coherent claim supported by complete experiments; applicable submission policies checked at that time. |

## Experiment design

### Comparisons

Core comparisons: aligned CRSDUN; CRSDUN with tuned static task weights; CRSDUN with a justified established balancing method; CRSDUN with matched corruption augmentation; proposed coupling alone; supporting balancing alone if used; combined method if warranted. Include an appropriate two-stage reconstruction plus segmentation pipeline and newer relevant methods identified by the review.

Match train/validation/test partitions, preprocessing, augmentation, initialization protocol, tuning budget and training budget. If crop sampling or class weighting changes, apply it to both baseline and proposed method or report it as a separate ablation. Do not attribute a data-pipeline repair to the proposed architecture.

### Measurement conditions

- Clean simulated measurements.
- Physically parameterized shot/read noise across a fixed severity grid.
- Mask shifts and calibration perturbations; distinguish known changed operators from unknown mismatch.
- Dispersion perturbations and unseen masks only after the operator is validated.
- Separate known corruption levels from held-out severities/types. Keep corruption seeds paired across methods.

For an unknown-mismatch experiment, generate observations with the perturbed operator but provide only the nominal operator to the reconstruction method. Giving it the true perturbed operator evaluates a different problem. Noise levels require an explicit intensity/exposure convention; our data are not uniformly bounded by one.

### Measurements and statistics

- Reconstruction: PSNR and SSIM with explicit dynamic-range conventions; spectral angle mapper and foreground-region spectral fidelity.
- Segmentation: foreground macro mIoU, macro F1, per-class IoU, real/fake confusion, and boundary quality where label quality permits.
- Reliability: calibration/error-detection performance and gate behavior, including minority classes and wrong-but-confident predictions.
- Efficiency: parameters, FLOPs with stated conventions, wall-clock inference, peak GPU memory, and training time on named hardware.
- Prefer at least three training seeds for key comparisons. Report variation across seeds separately from scene-level paired confidence intervals. Do not use individual pixels as independent statistical replicates.
- Select architectures, severity grids, metric conventions, and checkpoints using training/validation. Evaluate the fixed final protocol on held-out test data without tuning to it.

Avoid one arbitrary weighted score hiding a reconstruction–segmentation tradeoff. Report both axes; a useful result may dominate the baseline or achieve a clearly justified tradeoff. Set numerical go/no-go margins after measuring baseline variance, before final proposed-model evaluation.

## Data and compute constraints

The public252 dataset is adequate for development but is not currently an exact reproduction of the paper. Seek the authors' processed dataset, exact exclusions/splits, and spectral preprocessing through the public links; contact authors only with user authorization. If unavailable, publish an explicit public protocol and retrain all comparisons consistently. Do not claim to beat the paper by comparing unmatched numbers.

A second dataset must provide suitable hyperspectral data and segmentation labels. Reconstruction-only datasets cannot alone validate the joint task. If using a large remote-sensing cube, split by spatial regions with buffers to avoid patch overlap leakage and state the domain change. Simulated CASSI from measured HSI is not real captured CASSI.

The laptop's 4 GB GPU is useful for debugging. The observed five-stage diagnostic took approximately 20 seconds per attempted update; extrapolating to 202 batches suggests roughly an hour per epoch and weeks for a 500-epoch run, before repeated seeds and ablations. This is a planning estimate, not a full-training benchmark. Confirm server access, deadline and affordable compute before scheduling the full matrix. Use three-stage screening and matched budgets; larger final experiments need substantially more compute or a deliberately narrower scope.

## Immediate next work

1. Finish the paper/code/protocol comparison and related-work matrix before implementing a new research module.
2. Establish the original-data availability and compute plan while completing balanced overfit checks, explicit split selection and resumable baseline training.
3. Train a credible aligned control, then collect the failure-analysis evidence needed to accept or reject the reliability-conditioned cooperation hypothesis.

## Calendar to 30 November 2026

User-confirmed target: finish the paper by the end of November 2026. A university GPU may have approximately 16 GB memory; exact GPU model, GPU VRAM (rather than host RAM), access hours and job limits still need confirmation. No journal has been selected. The target below is a complete manuscript and reproducibility package, not a promise of acceptance or a reason to skip validation.

| Dates | Priority | Deliverable / decision |
| --- | --- | --- |
| 16–27 September | Paper-to-code audit, closest-work matrix, data/protocol decisions, balanced learning check, server setup, resumable training | Stable pipeline, documented dataset difference, measured training cost, shortlist of journals with supervisor |
| 28 September–11 October | Credible baseline runs, limited corruption and stage-wise failure analysis, establish variance | Baseline report and a specific failure hypothesis; stop architectural work if the baseline remains unreliable |
| 12–25 October | Minimal proposed method and decisive controls | Mid-October feasibility review; by 25 October freeze one main method or narrow/revise the claim if evidence is weak |
| 26 October–15 November | Main comparisons, key ablations, repeated seeds, robustness and external-data checks as feasible | Locked experiment tables; final held-out evaluation only after method/protocol selection |
| 16–30 November | Complete manuscript, supervisor revisions, figures, reproducibility and consistency checks | Journal-ready draft by 30 November if the evidence is sufficient; move submission rather than overclaim if it is not |

Write literature, protocol and methods sections continuously; the final fortnight is not the start of writing. Reserve server capacity early. Avoid simultaneously launching several architectural directions. Reassess the experiment budget after the first measured server epoch and checkpoint.

Pending confirmations: exact university GPU and access limits, supervisor's venue/scope preference, access to original processed data and real CASSI measurements. Select the journal by early October so scope and manuscript requirements can guide the remaining work.
