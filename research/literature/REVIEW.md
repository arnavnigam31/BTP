# Literature review — first pass

Last updated: 16 September 2026. Fourteen paper records; reading depth varies. Facts below are attributed reports, not independent replications. Interpretations and next steps are ours. Refer to the claims ledger for manuscript drafting.

| ID | Topic | Source | Reading depth |
| --- | --- | --- | --- |
| CRSDUN2026 | Core joint method | [Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf) | Main paper methods/experiments plus full supplement text reviewed; supplement Tables 3–7 visually checked |
| SSR2024 | Reconstruction backbone | [Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Improving_Spectral_Snapshot_Reconstruction_with_Spectral-Spatial_Rectification_CVPR_2024_paper.html) | Official abstract reviewed; full-text inspection pending |
| DPU2024 | Unfolding foundations | [Dual Prior Unfolding for Snapshot Compressive Imaging](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Dual_Prior_Unfolding_for_Snapshot_Compressive_Imaging_CVPR_2024_paper.pdf) | Targeted indexed full-text excerpt reviewed; metadata cross-checked against CRSDUN reference 45 |
| GST2022 | Measurement uncertainty | [Modeling Mask Uncertainty in Hyperspectral Image Reconstruction](https://arxiv.org/abs/2112.15362v4) | Author abstract plus targeted full-text Section 4.1, Tables 1–3 and mask-set appendix passages reviewed; derivation not fully checked |
| RDLUF2023 | Measurement mismatch | [Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging](https://arxiv.org/abs/2211.06891v3) | Author abstract reviewed; full-text inspection pending |
| DAUHST2022 | Degradation-aware unfolding | [Degradation-Aware Unfolding Half-Shuffle Transformer for Spectral Compressive Imaging](https://arxiv.org/abs/2205.10102v3) | Author abstract reviewed |
| FVgNET2022 | Dataset provenance | [Real-Time Hyperspectral Imaging in Hardware via Trained Metasurface Encoders](https://openaccess.thecvf.com/content/CVPR2022/papers/Makarenko_Real-Time_Hyperspectral_Imaging_in_Hardware_via_Trained_Metasurface_Encoders_CVPR_2022_paper.pdf) | Targeted full-text Section 4 and Fig. 5 caption reviewed; dataset paragraph visually checked on PDF page 5 |
| GRADNORM2018 | Task balancing | [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html) | Publisher metadata and full-text Sections 3.1–3.2 reviewed; experiment tables not audited |
| PCGRAD2020 | Gradient conflict | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782v4) | Author abstract reviewed |
| UNCERTAINTY2018 | Task weighting | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115v3) | Author abstract screened; detailed formulation pending |
| CALIBRATION2017 | Confidence calibration | [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Publisher abstract and bibliography reviewed |
| SEGNETMRI2018 | Adjacent joint inverse imaging | [Joint CS-MRI Reconstruction and Segmentation with a Unified Deep Network](https://arxiv.org/abs/1805.02165) | Author abstract reviewed |
| VLDUN2026 | Recent adjacent joint unfolding | [Vision-Language Controlled Deep Unfolding for Joint Medical Image Restoration and Segmentation](https://arxiv.org/abs/2601.23103) | Author abstract reviewed |
| STVIT2022 | Attention lineage | [Vision Transformer with Super Token Sampling](https://arxiv.org/abs/2211.11167v2) | Author abstract and version metadata reviewed |

## CRSDUN2026 — Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding

**Metadata:** He, Zijun; Wang, Ping; Wang, Xiaodong; Chen, Chang; Yuan, Xin. 2026. CVPR.

**Source:** [CRSDUN2026](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf); accessed 2026-09-16.

**Reading depth:** Main paper methods/experiments plus full supplement text reviewed; supplement Tables 3–7 visually checked.

- **CRSDUN2026-01 — reported fact:** Alternating reconstruction/segmentation unfolding and CASTA provide cross-task interactions. Locator: Sections 4.2–4.3.

- **CRSDUN2026-02 — reported fact:** Task balancing and segmentation sensitivity to noise/mask errors are identified as open issues. Locator: Section 6.

- **CRSDUN2026-03 — reported fact:** Stage-specific degradation estimators and shot-noise training are already included. Locator: Eq. 23; Section 5.4.


**Relevance (our interpretation):** Principal control. Our method must differ from existing cooperation and degradation handling.

**Next verification:** Audit paper-to-code equivalence and resolve processed-data provenance. See supplement facts in ledger.

## SSR2024 — Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification

**Metadata:** Zhang, Jiancheng; Zeng, Haijin; Chen, Yongyong; Yu, Dengxiu; Zhao, Yin-Ping. 2024. CVPR.

**Source:** [SSR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Improving_Spectral_Snapshot_Reconstruction_with_Spectral-Spatial_Rectification_CVPR_2024_paper.html); accessed 2026-09-16.

**Reading depth:** Official abstract reviewed; full-text inspection pending.

- **SSR2024-01 — reported fact:** WSSA captures spectral information; ARB addresses spatial degradation through alignment. Locator: Official abstract.


**Relevance (our interpretation):** CRSDUN builds on these reconstruction components. Distinguish ARB's imaging degradation from our loader orientation bug.

**Next verification:** Inspect forward model, ARB, training protocol and code before implementing a matched two-stage baseline.


Archived PDF: [source](sources/SSR2024.pdf). Downloading does not upgrade review depth.

## DPU2024 — Dual Prior Unfolding for Snapshot Compressive Imaging

**Metadata:** Zhang, Jiancheng; Zeng, Haijin; Cao, Jiezhang; Chen, Yongyong; Yu, Dengxiu; Zhao, Yin-Ping. 2024. CVPR.

**Source:** [DPU2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Dual_Prior_Unfolding_for_Snapshot_Compressive_Imaging_CVPR_2024_paper.pdf); accessed 2026-09-16.

**Reading depth:** Targeted indexed full-text excerpt reviewed; metadata cross-checked against CRSDUN reference 45.

- **DPU2024-01 — reported fact:** Uses CAVE for training and ten KAIST scenes for simulated evaluation. Locator: Section 4.1.

- **DPU2024-02 — reported fact:** Sparse attention uses learned threshold filtering. Locator: Eqs. 14–16.


**Relevance (our interpretation):** Prior-module and attention-thresholding precedent; generic filtering is not enough to distinguish our proposal.

**Next verification:** Read full objective/derivation; its reconstruction scores are not FVgNET joint-task scores.

## GST2022 — Modeling Mask Uncertainty in Hyperspectral Image Reconstruction

**Metadata:** Wang, Jiamian; Zhang, Yulun; Yuan, Xin; Meng, Ziyi; Tao, Zhiqiang. 2022. ECCV (verified via author arXiv record).

**Source:** [GST2022](https://arxiv.org/abs/2112.15362v4); accessed 2026-09-16.

**Reading depth:** Author abstract plus targeted full-text Section 4.1, Tables 1–3 and mask-set appendix passages reviewed; derivation not fully checked.

- **GST2022-01 — reported fact:** Models mask uncertainty with variational Bayesian learning, a graph-based self-tuning network and bilevel optimization. Locator: Abstract.

- **GST2022-02 — reported fact:** Distinguishes same-mask, one-to-many and many-to-many evaluation; one-to-many testing uses unseen masks and 100 trials. Locator: Section 4.1; Table 2; supplementary scenario discussion.


**Relevance (our interpretation):** Direct novelty constraint on uncertainty/miscalibration claims; distinguish hardware uncertainty from cross-task trust.

**Next verification:** Finish uncertainty derivation and inspect which mask/operator information is supplied during evaluation.


Archived PDF: [source](sources/GST2022.pdf). Downloading does not upgrade review depth.

## RDLUF2023 — Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging

**Metadata:** Dong, Yubo; Gao, Dahua; Qiu, Tian; Li, Yuyan; Yang, Minxi; Shi, Guangming. 2023. CVPR (verified via author arXiv record).

**Source:** [RDLUF2023](https://arxiv.org/abs/2211.06891v3); accessed 2026-09-16.

**Reading depth:** Author abstract reviewed; full-text inspection pending.

- **RDLUF2023-01 — reported fact:** Learns residual degradation to reduce sensing-model mismatch and uses a spatial/spectral prior transformer. Locator: Abstract.


**Relevance (our interpretation):** Operator correction is established; ours must explain a distinct contribution in joint-task cooperation.

**Next verification:** Check learned operator inputs and real-data protocol; separate correction from confidence conditioning.

## DAUHST2022 — Degradation-Aware Unfolding Half-Shuffle Transformer for Spectral Compressive Imaging

**Metadata:** Cai, Yuanhao; Lin, Jing; Wang, Haoqian; Yuan, Xin; Ding, Henghui; Zhang, Yulun; Timofte, Radu; Van Gool, Luc. 2022. NeurIPS (verified via author arXiv record).

**Source:** [DAUHST2022](https://arxiv.org/abs/2205.10102v3); accessed 2026-09-16.

**Reading depth:** Author abstract reviewed.

- **DAUHST2022-01 — reported fact:** Estimates unfolding-control parameters from compressed measurements and masks; uses a half-shuffle transformer. Locator: Abstract.


**Relevance (our interpretation):** Adaptive unfolding parameters already exist and cannot alone constitute the proposed contribution.

**Next verification:** Read parameter estimator equations and compare directly with CRSDUN Eq. 23.

## FVgNET2022 — Real-Time Hyperspectral Imaging in Hardware via Trained Metasurface Encoders

**Metadata:** Makarenko, Maksim; Burguete-Lopez, Arturo; Wang, Qizhou; Getman, Fedor; Giancola, Silvio; Ghanem, Bernard; Fratalocchi, Andrea. 2022. CVPR.

**Source:** [FVgNET2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Makarenko_Real-Time_Hyperspectral_Imaging_in_Hardware_via_Trained_Metasurface_Encoders_CVPR_2022_paper.pdf); accessed 2026-09-16.

**Reading depth:** Targeted full-text Section 4 and Fig. 5 caption reviewed; dataset paragraph visually checked on PDF page 5.

- **FVgNET2022-01 — reported fact:** Introduces a metasurface-based imaging system and a labeled hyperspectral segmentation dataset. Locator: Abstract.


**Relevance (our interpretation):** Original dataset provenance. Its hardware is not the same acquisition model as simulated CASSI.

**Next verification:** Reconcile source dataset description with public mirrors and author-preprocessed CRSDUN data; verify original license.


Archived PDF: [source](sources/FVgNET2022.pdf). Downloading does not upgrade review depth.

## GRADNORM2018 — GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks

**Metadata:** Chen, Zhao; Badrinarayanan, Vijay; Lee, Chen-Yu; Rabinovich, Andrew. 2018. Proceedings of the 35th International Conference on Machine Learning.

**Source:** [GRADNORM2018](https://proceedings.mlr.press/v80/chen18a.html); accessed 2026-09-16.

**Reading depth:** Publisher metadata and full-text Sections 3.1–3.2 reviewed; experiment tables not audited.

- **GRADNORM2018-01 — reported fact:** Adapts task weights through gradient-magnitude balancing. Locator: Publisher abstract.

- **GRADNORM2018-02 — reported fact:** Targets gradient norms using relative training rates, treats targets as constants when updating weights, and renormalizes task weights. Locator: Sections 3.1–3.2; Eqs. 1–2.


**Relevance (our interpretation):** Essential control if we propose adaptive loss weighting. Record extra backward-pass and memory costs.

**Next verification:** Map shared parameters in CRSDUN; measure actual overhead rather than transferring the paper's timing.


Archived PDF: [source](sources/GRADNORM2018.pdf). Downloading does not upgrade review depth.

## PCGRAD2020 — Gradient Surgery for Multi-Task Learning

**Metadata:** Yu, Tianhe; Kumar, Saurabh; Gupta, Abhishek; Levine, Sergey; Hausman, Karol; Finn, Chelsea. 2020. NeurIPS (verified via author arXiv record).

**Source:** [PCGRAD2020](https://arxiv.org/abs/2001.06782v4); accessed 2026-09-16.

**Reading depth:** Author abstract reviewed.

- **PCGRAD2020-01 — reported fact:** Projects a task gradient against another conflicting task gradient to reduce interference. Locator: Abstract.


**Relevance (our interpretation):** Separates gradient-direction conflict from unequal loss magnitudes; a candidate control, not our invention.

**Next verification:** Read projection/order details and distinguish shared from task-exclusive parameter handling.

## UNCERTAINTY2018 — Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

**Metadata:** Kendall, Alex; Gal, Yarin; Cipolla, Roberto. 2018. CVPR; verify proceedings metadata before submission.

**Source:** [UNCERTAINTY2018](https://arxiv.org/abs/1705.07115v3); accessed 2026-09-16.

**Reading depth:** Author abstract screened; detailed formulation pending.

- **UNCERTAINTY2018-01 — reported fact:** Studies relative weighting of regression and classification objectives using task uncertainty. Locator: Abstract.


**Relevance (our interpretation):** Compare against uncertainty weighting; do not conflate a global task weight with spatially calibrated prediction reliability.

**Next verification:** Verify exact likelihood assumptions, regularization terms and published metadata.

## CALIBRATION2017 — On Calibration of Modern Neural Networks

**Metadata:** Guo, Chuan; Pleiss, Geoff; Sun, Yu; Weinberger, Kilian Q.. 2017. Proceedings of the 34th International Conference on Machine Learning.

**Source:** [CALIBRATION2017](https://proceedings.mlr.press/v70/guo17a.html); accessed 2026-09-16.

**Reading depth:** Publisher abstract and bibliography reviewed.

- **CALIBRATION2017-01 — reported fact:** Finds miscalibration in modern classifiers and evaluates temperature scaling. Locator: Publisher abstract.


**Relevance (our interpretation):** Confidence must be evaluated, not assumed trustworthy. Classification evidence does not establish dense CASSI calibration.

**Next verification:** Specify validation-only fitting, foreground/class-conditioned metrics and shifted-distribution tests.

## SEGNETMRI2018 — Joint CS-MRI Reconstruction and Segmentation with a Unified Deep Network

**Metadata:** Sun, Liyan; Fan, Zhiwen; Huang, Yue; Ding, Xinghao; Paisley, John. 2018. arXiv record; archival venue not checked.

**Source:** [SEGNETMRI2018](https://arxiv.org/abs/1805.02165); accessed 2026-09-16.

**Reading depth:** Author abstract reviewed.

- **SEGNETMRI2018-01 — reported fact:** Combines MRI reconstruction and segmentation with shared reconstruction encoders and fine-tuning. Locator: Abstract.


**Relevance (our interpretation):** Broad joint reconstruction–segmentation is not new. Any first-of-kind claim must be precisely scoped.

**Next verification:** Read optimization and evaluation details before drawing architectural comparisons.

## VLDUN2026 — Vision-Language Controlled Deep Unfolding for Joint Medical Image Restoration and Segmentation

**Metadata:** Chen, Ping; Huang, Zicheng; Wang, Xiangming; Liu, Yungeng; Liang, Bingyu; Zeng, Haijin; Chen, Yongyong. 2026. arXiv preprint; peer-reviewed status not established.

**Source:** [VLDUN2026](https://arxiv.org/abs/2601.23103); accessed 2026-09-16.

**Reading depth:** Author abstract reviewed.

- **VLDUN2026-01 — reported fact:** Proposes coupled medical restoration–segmentation unfolding with a frequency-aware Mamba mechanism. Locator: Abstract.


**Relevance (our interpretation):** Recent adjacent precedent; avoid presenting joint unfolding or a backbone change as sufficient novelty.

**Next verification:** Inspect equations, language control, training supervision and actual publication status.

## STVIT2022 — Vision Transformer with Super Token Sampling

**Metadata:** Huang, Huaibo; Zhou, Xiaoqiang; Cao, Jie; He, Ran; Tan, Tieniu. 2022. arXiv v2 revised 2024; archival venue not verified.

**Source:** [STVIT2022](https://arxiv.org/abs/2211.11167v2); accessed 2026-09-16.

**Reading depth:** Author abstract and version metadata reviewed.

- **STVIT2022-01 — reported fact:** STA samples super tokens, attends among them, then maps them back to visual tokens. Locator: Abstract.


**Relevance (our interpretation):** Direct conceptual lineage for CASTA. Super-token aggregation itself is not a new contribution.

**Next verification:** Read sparse association learning and isolate CRSDUN's cross-task modification.
