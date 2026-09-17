# Search log and reading queue

Date: 2026-09-16. Discovery tools: web search, official CVF/PMLR pages, author arXiv records, CRSDUN references and supplementary material. The log records query families actually used across this review and the immediately preceding planning pass. It is a targeted scoping review, not a PRISMA/systematic-search claim; no exhaustive hit count or citation-coverage claim is made.

## Search families

| Query / route | Purpose | Outcome |
| --- | --- | --- |
| Exact CRSDUN title; official CVF main/supplement | Anchor and follow-up discovery | Main paper and five-page supplement obtained; no exhaustive forward-citation review completed. |
| CRSDUN uncertainty adaptive reconstruction segmentation CASSI | Candidate overlap | Mask-uncertainty and degradation-learning prior work identified. |
| CASSI joint reconstruction segmentation uncertainty cooperative unfolding; spectral segmentation task unfolding | Joint-task and adjacent methods | SegNetMRI and VL-DUN included as adjacent precedents. |
| Exact SSR and Dual Prior Unfolding titles on CVF | Backbone and proximal-prior lineage | Official SSR abstract/PDF and DPU full-text excerpt checked. |
| FVgNET on Nature; exact title of CRSDUN reference 23 | Dataset provenance | Original CVPR 2022 paper obtained; later FVgNET-video paper found but not yet reviewed. |
| GradNorm; Gradient Surgery for Multi-Task Learning on arXiv | Optimization controls | GradNorm and PCGrad records checked. |
| Uncertainty weighting; On Calibration of Modern Neural Networks | Reliability controls | Kendall et al. and Guo et al. included. |
| Vision Transformer with Super Token Sampling | CASTA lineage | Author arXiv v2 checked; final archival metadata unresolved. |

Only primary-source claims enter the ledger. Search-engine timestamps are not treated as publication dates. Third-party summaries were discovery aids, not evidentiary sources. BibTeX uses checked proceedings metadata where available and conservative arXiv entries otherwise.

## Prioritized remaining reading

1. **Dataset releases:** original FVgNET supplement/repository and author-preprocessed CRSDUN data. Verify mask counts, wavelength calibration, mapping of unknown/starfruit, normalization, original licenses and exact excluded scene IDs.
2. **Closest uncertainty papers:** finish GST and RDLUF methods and evaluation protocols; RDLUF's requested arXiv PDF returned HTTP 406 and needs a publisher-source fallback.
3. **Shared-parameter optimization:** full PCGrad/Kendall and GradNorm experiments; determine applicability to CRSDUN's coupled branches before implementation.
4. **Reliability-conditioned task interaction:** focused search on uncertainty-gated restoration/segmentation, confidence-aware feature fusion, adaptive-depth joint inverse problems, and negative transfer. This is still a major gap in novelty coverage.
5. **Recent work and forward citations:** 2025–2026 CASSI/SCI reconstruction, joint-task unfolding and CRSDUN citing papers. Update before method freeze and submission.
6. **Baselines already in CRSDUN supplement:** PADUT, SPECAT and RCUMP; inspect original papers rather than treating CRSDUN's reimplementation scores as original-author results.
7. **External validity:** suitable labeled HSI datasets and real CASSI data access. Dataset names alone do not establish an appropriate joint-task benchmark.

## Retrieval status

Five of six attempted new PDF downloads succeeded (SSR, GST, GradNorm, original FVgNET, CRSDUN supplement). RDLUF failed with HTTP 406; its verified abstract is retained, and full-text claims are not made from the failed retrieval. The CRSDUN main PDF was already present. Checksums are in `download_manifest.json`.
