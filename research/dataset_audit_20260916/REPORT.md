# Dataset completeness and preprocessing audit

Checked 16 September 2026. All dataset operations were read-only. Reports are stored separately. Audit script: [audit_dataset.py](../audit_dataset.py); machine-readable evidence: [audit.json](audit.json) and [per-scene checks and hashes](scenes.json).

## Verdict

**The local mask-labeled public subset is complete relative to its metadata and internally valid. It is not the entire original FVgNET collection and is not a verified copy of the paper's processed CRSDUN dataset.** It is suitable for controlled development with the orientation correction enabled. Preprocessing provenance and the intensity convention need resolution before claiming an exact paper reproduction or reporting a defensible comparative baseline.

## Completeness

| Component | Count / result |
| --- | --- |
| Local public metadata | 315 distinct scene IDs |
| Metadata rows with segmentation masks | 252 |
| Metadata rows without masks | 63; their cubes are not in the local source34 directory |
| Source visible cubes | 252, each 512×512×34 |
| Candidate converted cubes | 252, each 512×512×28 |
| Public RGB labels, raw two-channel labels, mapped RGB labels, mapped class-ID arrays | 252 of each; exact labeled-ID set matches |
| Development split | 202 train / 25 validation / 25 test; disjoint IDs and exact union |
| Missing or unreadable required labeled files | None found |
| Invalid mapped IDs / RGB palette / raw-to-mapped conversions | None found |
| Nonfinite cube/label values | None found |
| Expected classes absent from a split | None; IDs 0–22 represented in each |
| Exactly duplicate candidate cubes | None by full-array SHA-256 |
| Original author-preprocessed download directory | Empty; repository supplies only a dataset-download README, no author split CSVs locally |

The original FVgNET project describes 317 scenes, with segmentation masks for about 80%. Our metadata lists 315, exactly 252 of which have masks. The two-scene difference cannot be identified without a complete upstream scene manifest. The 63 metadata scenes without masks are not missing training labels from our chosen subset; they are outside its supervised scope. [Original author dataset description](https://github.com/makamoa/hyplex#fvgnet-spectral-imaging-dataset).

CRSDUN reports 256 training and 50 test scenes after excluding 11 scenes. We cannot infer that downloading 54 arbitrary additional scenes will recover that protocol: the memberships, exclusions and processed release must be checked. [Paper, Section 5.1](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf); [author repository and dataset link](https://github.com/zjhe02/CRSDUN).

## Preprocessing status

| Operation | Status | Required action |
| --- | --- | --- |
| Spatial orientation | In-memory correction implemented, visually verified on four training scenes | Use `--transpose_image` consistently for this candidate dataset. This audit does not establish pixel-perfect registration for every annotation. |
| 34 → 28 spectral conversion | Already applied; checked against all source cubes | Do not repeat it. Maximum absolute discrepancy from manifest-defined linear interpolation: 1.78e-15. |
| BK7 target grid | Internally matches equal spacing in refractive-index coordinates; deviation 2.22e-16 | Verify exact wavelengths/optical convention against author-processed data. Internal numerical consistency is not physical/author-protocol validation. |
| Label conversion | Raw IDs and real/fake flags match mapped IDs and palette for every scene | Already done. Six scenes contain starfruit; mapping it to unknown is an explicit project choice, not verified author behavior. |
| Radiometric normalization | No further normalization/clipping in our 34→28 conversion; upstream source convention unresolved | Determine what scaling the public 34-band cubes already use. Do not apply per-image min–max scaling or clip automatically. |
| Tensor dtype | Cubes stored as float64; training converts inputs to float32 and labels to integer tensors | Existing training conversion is sufficient; file conversion is optional storage optimization, not a scientific requirement. |
| Cropping and flipping | Existing paired transformations pass coordinate-registration tests | Preserve identical image/label transforms. Label resizing, if introduced, must preserve discrete classes. |
| Split selection and provenance | Development CSVs fixed and disjoint | Preserve them. Explicit held-out split selection in evaluation code is a separate remaining task. |

### Intensity scale deserves a specific check

Training values range from 0 to 8.814; about 3.41% of training values exceed one. Validation also has values above one. The original demo loads from `normalizedHSI`, but that does not prove which radiometric processing the downloaded 34-band mirror received. Values above one do not alone demonstrate corruption or imply that clipping is correct.

Current `src/CRSDUN/utils.py`, `Metrics_Rec.add_batch`, uses `data_range=1.0` for PSNR and SSIM. We must justify that reference scale against the radiometric convention. Changing the metric range or image scale changes reported scores and must be applied consistently to every comparison. Any fitted normalization must use training data only and be fixed before validation/test evaluation. No normalization was fitted or changed in this audit.

## Portability and limitations

The development directory contains symlinks to the converted cubes and labels under `fvgnet_public`; both targets exist. It is not a standalone dataset copy. Moving it to the university server must preserve valid targets or package their contents deliberately. No files were moved or deleted here.

Full-array hashes rule out exact duplicates only, not repeated objects, adjacent acquisitions or near-duplicate scenes. Without upstream checksums, the audit proves internal integrity rather than byte-for-byte identity with the original release. No model was evaluated. Test files were opened for integrity and conversion checks only; this expands on earlier audits that had left their pixels unread.

## Next action

Before another long training run, verify the original processed-data availability and intensity convention. If the author release remains inaccessible, freeze a transparent public252 preprocessing specification, retain the explicit starfruit decision and candidate wavelength grid, and retrain every baseline under that same specification. Further downloads of mask-free scenes are not required merely to make the existing supervised subset complete.

## Follow-up resolution, 16 September

The operational choices above are now fixed for public252-v1; see [resolution](../preprocessing_resolution/RESOLUTION.md). Exact author-release equivalence remains unverified. Original audit observations are retained unchanged.
