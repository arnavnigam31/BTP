# Public252 pipeline check — 16 September 2026

## Confirmed alignment issue

The public FVgNET candidate image cubes have their two spatial axes reversed relative to the mapped label PNGs. In four inspected training scenes, transposing the image from `(axis0, axis1, band)` to `(axis1, axis0, band)` aligns object silhouettes with the labels. The original public label PNGs also show this orientation; the issue is not introduced by the CRSDUN-like color mapping.

The dataset files and symlinks have not been changed. The loader correction is explicit and in memory. Use `--transpose_image` for this public252 candidate dataset. The default remains unchanged for other datasets.

The earlier `public252_smoke_e1` checkpoint trained on misregistered images and labels. Keep it as a historical smoke-test artifact, but do not use it as an aligned baseline or resume the corrected experiment from it.

## Audit

- 202 training, 25 validation, and 25 test scene IDs; no duplicate IDs within splits or overlap across them; all named files exist.
- Training and validation image/label dimensions agree, cubes have 28 bands, image values are finite, and label colors belong to the expected palette.
- All 22 foreground classes appear in both training and validation.
- Foreground is 5.162% of full training images and 4.800% of validation images.
- Across 1,616 sampled training crops, median foreground is 11.783%; no crop was empty. This is a sample, not an exhaustive guarantee about every crop.
- Image values range from 0 to 8.814 in training and 0 to 3.136 in validation. Values were not clipped or normalized by this audit.
- Test pixels were not read. Split checks establish separation of scene IDs, not separation of related acquisitions or duplicate content.

## Code checks

The batch and stage loops now use distinct counters. Progress is reported at every tenth batch and the final batch. The in-memory transpose option is passed through single-GPU training, DDP training, and the existing evaluation entry point.

Run the CPU regression checks from the CRSDUN directory:

```bash
/home/arnav/miniforge3/envs/cassi/bin/python verify_public252_pipeline.py \
  --root /home/arnav/btp-cassi/data/crsdun_public252_dev
```

## Learning diagnostic

The unaligned diagnostic was interrupted after inspecting the overlays. The corrected diagnostic starts from scratch on two fixed, foreground-rich training crops, using the five-stage model, fixed measurement mask, Adam at 0.0004, and the original loss weights (reconstruction 1, segmentation 0.0001). It alternates the crops for 40 attempted optimizer steps without augmentation or learning-rate decay.

This is a memorization diagnostic, not a baseline benchmark. Its foreground mean IoU is computed only over foreground classes present in these two crops; it is not directly comparable to the original full validation mean IoU.

Mixed-precision loss scaling may skip early optimizer steps with nonfinite gradients. The saved optimizer state must be checked to count effective updates; attempted steps alone are not evidence of learning.

Outputs are in `/home/arnav/btp-cassi/runs/public252_aligned_diagnostic_20260916/`.

### Completed result

The run finished in approximately 13.7 minutes of GPU diagnostic time. Adam recorded 39 effective updates out of 40 attempts; the first overflow caused loss scaling to fall from 65,536 to 32,768. All saved model tensors are finite.

| Metric on the two training crops | Initial | After 40 attempts |
| --- | ---: | ---: |
| Final-stage reconstruction MSE | 0.61008 | 0.02223 |
| Final-stage segmentation cross-entropy | 3.15308 | 0.83468 |
| Mean IoU across the six present foreground classes | 1.03% | 9.77% |
| Foreground pixel recall (correct class) | 4.21% | 45.62% |

Final real-plant IoU is 49.79%, and fake-grape IoU is 8.84%; the other four present foreground classes have zero IoU. Predictions visibly follow object shapes but overuse the real-plant class. This establishes that the corrected pipeline can learn some foreground, not that the model has successfully memorized all classes or generalizes.

The original dataset remains unchanged. No GitHub operations were performed during this check. The progress/alignment code changes and diagnostic files are currently uncommitted.

## Follow-up before full training

1. Extend the small-sample check with more balanced foreground classes and more updates. Retain the original loss weights as the control; avoid interpreting the current dominant-class learning as a passed multi-class overfit test.
2. If the remaining classes still do not learn, isolate segmentation loss/gradient behavior before launching a long run.
3. Add resumable baseline checkpoints and perform an aligned multi-epoch pilot from scratch.
4. Before held-out evaluation, give `test.py` explicit split selection: it currently calls the shared validation loader, so its output named “Test stats” is validation performance.
5. Continue to describe this as a public252 development experiment with candidate spectral preprocessing, not an exact original-paper reproduction.
