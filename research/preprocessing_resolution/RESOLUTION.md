# Public252 preprocessing decisions — 16 September 2026

## Outcome and limits

The operational preprocessing questions are resolved for **public252-v1**, a controlled public-subset benchmark. This is not an exact replication of CRSDUN's 306-scene processed dataset. No images, labels, CSVs, or dataset manifests were changed. No model training or test evaluation was performed in this step.

The frozen machine-readable specification is [public252-v1.json](public252-v1.json). It includes the exact 28 wavelengths, complete raw-label mapping, split sizes, and SHA-256 hashes of seven existing metadata files. [run_public252.py](run_public252.py) validates the specification and prepares training with the orientation correction. It defaults to checking only; training requires `--train`. Future comparisons must use the same protocol and retrain baselines.

## 1. Intensity: retain distributed values, no clipping or extra scaling

The original Hyplex segmentation notebook defines `ImageNormalize` and `Clamp` classes but **does not include either in its active `prep_loaders` transform composition**. That composition uses cropping, tensor conversion and flipping. Its loader reads the distributed visible cube directly. The reconstruction notebook shows linear spectral interpolation and a commented save to the visible directory, with no additional scale in that conversion cell. These are evidence for keeping the distributed numerical scale; they do not establish its physical radiometric calibration or prove that our download is byte-identical to the authors' data.

Decision: use identity intensity preprocessing, retain values above one, convert tensors to float32, and do not apply per-image maximum division, min-max normalization, percentile clipping, or reference-board renormalization. No normalization statistics are fitted on train, validation or test data.

CRSDUN's metric implementation uses `data_range=1.0` for PSNR and SSIM. Retain this fixed reference and report it explicitly as **PSNR_ref1 / SSIM_ref1** in our research tables. This does not assert that targets lie in [0,1]. For reference amplitude one, PSNR is -10 log10(MSE). The existing training log headings remain PSNR/SSIM; the saved protocol defines their meaning. Changing the reference to each image's maximum would change the benchmark and is not permitted within v1. Raw-domain MSE should accompany reconstruction results when assembling the evaluation pipeline. Synthetic targets spanning [0,2] with an additive 0.1 error produced the expected 20 dB; SSIM matched the explicit unit-reference implementation.

## 2. Spatial orientation: keep the in-memory transpose

The original Hyplex `HySpecSegmentation.read_image_label` explicitly calls `np.load(imagefile).transpose(1,0,2)`. This independently supports the correction previously verified visually on four of our training scenes. The v1 launcher always passes `--transpose_image`. It checks one actual training cube against its transposed source without altering values. It does not claim exhaustive pixel-perfect annotation registration.

## 3. Starfruit: other/unknown while preserving real/fake

The original FVgNET demo names raw fruit ID 12 as starfruit and ID 10 as unknown. Its `map_labels` function sends object names missing from `used_classes` to real/fake unknown. CRSDUN's published 23-class palette has real/fake unknown but no named starfruit class. We therefore retain raw ID 12 -> 21 (real) / 22 (fake), alongside original unknown ID 10. This is a source-supported adaptation of the original fallback convention to CRSDUN's taxonomy. The older demo's numerical unknown IDs differ, so its IDs must not be copied verbatim.

This resolves our mapping choice, not proof of CRSDUN's unpublished conversion. In manuscript text and class tables identify these classes as **other/unknown (including starfruit)**. Do not invent a new 25-class task or discard six scenes merely to guess author behavior.

## 4. Spectral grid: freeze the explicit candidate grid

The paper specifies 28 bands over 400–730 nm according to BK7 dispersion but does not give the exact target wavelengths in the inspected paper, supplement or code. The original Hyplex reconstruction notebook supplies the source 34-band `linspace(400,730,34)` convention. Our audit independently verified all 252 conversions against the existing manifest (maximum error 1.78e-15) and equal spacing in N-BK7 refractive index (2.22e-16 numerical deviation).

Decision: retain the existing 28 wavelengths and linear interpolation, recorded in full in the frozen specification. Describe it as an **equal-refractive-index N-BK7 approximation**. Equal refractive-index increments are not proof of exact prism ray-traced/angular dispersion or author wavelength calibration. Do not reinterpolate the already converted cubes. A later author grid comparison must create a new protocol version and new results, rather than silently replace v1.

## Verification and safeguards

- Preflight passed on all 252 image/label headers and frozen metadata; split counts 202/25/25 and disjoint membership confirmed.
- Actual loader check passed on one training scene: transpose applied, intensity values unchanged.
- Synthetic PSNR/SSIM checks passed above unit intensity.
- The launcher refuses changed frozen metadata and reuse of an existing output directory. It records the protocol with training output.
- The previous full-pixel audit remains [the integrity evidence](../dataset_audit_20260916/REPORT.md). Current preflight is not a replacement for a full rehash of image contents after transfer.
- The existing standalone `test.py` split-selection issue remains separate work; this launcher invokes training with the established validation loader, not held-out evaluation.

## Exact-paper equivalence: external evidence still required

The official Baidu page returned HTTP 200 with a published-code prompt. Browser access was then rejected by the browser site-safety policy; no workaround was attempted. No author files were downloaded or inspected. To close exact-equivalence questions we need the author train/test CSVs, exact 28 wavelengths or conversion script, intensity-processing description, and raw-to-23-class conversion (especially starfruit). A matched author cube and label can help verify orientation and scale, but a single sample cannot establish all-scene identity. No author contact message was sent.

## Primary evidence

Hyplex checkout `187cd9394dae08beb883dbc6bab50a6d3f317c66`. Code cells archived without notebook output images:

- [HyplexSegDemo source](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/HyplexSegDemo.ipynb): ImageNormalize/Clamp definitions; read_image_label; active prep_loaders composition. [Author notebook](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/HyplexSegDemo.ipynb).
- [FVgNETDemo source](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/FVgNETDemo.ipynb): raw fruit dictionary, map_labels fallback, normalizedHSI reader. [Author notebook](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/FVgNETDemo.ipynb).
- [HyplexRecDemo source](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/HyplexRecDemo.ipynb): source visible wavelength grid and interpolation cell. [Author notebook](https://github.com/makamoa/hyplex/blob/187cd9394dae08beb883dbc6bab50a6d3f317c66/jupyter/HyplexRecDemo.ipynb).
- [CRSDUN paper](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf), Section 5.1; [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/He_Joint_Spectral_Image_CVPR_2026_supplemental.pdf), Table 2 palette; [author repository](https://github.com/zjhe02/CRSDUN), dataset.py and utils.py.

## Paper-ready methods wording

We construct a 252-scene labeled public FVgNET development benchmark with fixed 202/25/25 training/validation/test partitions. Distributed 34-band visible cubes are linearly interpolated to 28 wavelengths spanning 400–730 nm, equally spaced in N-BK7 refractive index. Cube spatial axes are transposed to match annotations. Intensities are retained without clipping or additional normalization. Labels use a 23-class taxonomy, mapping starfruit into the corresponding real/fake other category. Reconstruction metrics use a fixed reference amplitude of one. This protocol differs from the processed 256/50 training/test dataset reported by CRSDUN; direct numerical comparison with its published scores is not made.
