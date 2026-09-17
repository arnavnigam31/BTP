# BTP / CRSDUN: university GPU handoff

This is the CRSDUN code repository extended for our public252-v1 benchmark. Original attribution and architecture remain in README.md. Do not compare public252 scores directly with the paper's different dataset split.

## Clone and install

```bash
git clone --branch btp-public252-baseline https://github.com/arnavnigam31/BTP.git
cd BTP
conda create -n cassi python=3.10 -y
conda activate cassi
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-btp.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"
```

These torch/torchvision versions are the actual tested local installation, not a claim that they suit every server driver. Verify package availability and driver compatibility on the server. If that pinned CUDA build is unavailable/incompatible, install a matched PyTorch/torchvision CUDA build appropriate for the server and rerun all checks below. Do not install the old full pip-freeze file: it contains a machine-local package URL and unrelated CUDA tooling. No matplotlib is required by evaluation now.

## Transfer the dataset separately

Git contains code, protocol metadata and research records. It does not contain the approximately 14 GB image dataset, trained weights or experiment outputs. From the laptop's WSL terminal, substitute the actual university SSH account and destination:

```bash
rsync -aL --info=progress2 /home/arnav/btp-cassi/data/crsdun_public252_dev/ USER@SERVER:/ABSOLUTE/DATA/PATH/public252/
```

`-L` copies the contents of the two dataset symlinks, making the destination self-contained. No deletion option is used. The destination must contain `visible_28/`, `labels/`, train/val/test/all CSVs and the three preprocessing/split manifests. Keep enough space for data, checkpoints and optional reconstructed test cubes.

On the server:

```bash
export DATA_ROOT=/ABSOLUTE/DATA/PATH/public252
python run_public252.py --root "$DATA_ROOT"
python verify_dataset_hashes.py --root "$DATA_ROOT"
python verify_public252_pipeline.py --root "$DATA_ROOT"
python verify_portability.py --root "$DATA_ROOT"
python verify_evaluation.py
python gpu_smoke.py --root "$DATA_ROOT" --out exp/server_gpu_smoke
```

The full cube hash check reads all 252 cubes for transfer integrity only, not model evaluation. It also verifies label file hashes and format/palette. Preflight checks frozen metadata, headers and the loader/metrics convention. Keep these distinctions in the experiment record.

## Learning diagnostic, then baseline

```bash
python diagnostic_public252.py --root "$DATA_ROOT" --out exp/server_fixed_crop_diagnostic --transpose-image --steps 200
```

The earlier 40-attempt diagnostic learned two dominant classes but did not pass multiclass overfitting. This remains a scientific check to complete; the CUDA smoke test below is only an infrastructure test. Inspect loss curves, predictions and per-class behavior; do not assert success merely because training runs. The diagnostic reads training/validation files but does not evaluate test pixels.

When ready, launch the baseline in a persistent server session such as tmux:

```bash
python run_public252.py --root "$DATA_ROOT" --train --epochs 500 --batch-size 1 --name public252_v1_baseline
```

Start with batch size one and verify actual memory use. Full 512x512 validation can use more memory than a 256x256 training crop. A 16 GB GPU has not been tested here. To measure an initial epoch without changing the 500-epoch cosine schedule, start the intended run, wait for the first complete epoch/last.pt, then interrupt if necessary. Resume redoes an interrupted epoch from its last completed boundary.

## Resume and checkpoint selection

```bash
python run_public252.py --root "$DATA_ROOT" --train --epochs 500 --batch-size 1 --name public252_v1_baseline --resume exp/CRSDUN/public252_v1_baseline/model/last.pt
```

Keep total epochs, batch size, learning rate, worker count, seed, measurement settings, orientation and frozen metadata unchanged. Dataset location may change; content identity and measurement mask must match. `last.pt` saves model, Adam, cosine scheduler, AMP scaler, epoch, best scores, RNG and configuration. It supports epoch-boundary continuation, not exact mid-epoch continuation. Cross-hardware bitwise equality is not promised.

`best_iou.pth` and `best_psnr.pth` are weights-only checkpoints selected on validation. Foreground mIoU is the recommended primary segmentation selection rule; report reconstruction at that same checkpoint, and identify any separately PSNR-selected result. The upstream metric averages all 22 foreground IDs, with absent IDs scored zero and epsilon 1e-4. Scores are fractions, not percentages; metrics are now retained at full precision before aggregation and checkpoint selection (the earlier upstream rounding has been removed). PSNR/SSIM use reference amplitude one, not per-image maxima. Each validation epoch writes reconstruction and per-class segmentation CSVs.

## Evaluate deliberately

```bash
python test.py --data_root "$DATA_ROOT" --transpose_image --eval_split val --batch_size 1 --name public252_v1_baseline --pretrained_model_path exp/CRSDUN/public252_v1_baseline/model/best_iou.pth
# Only after model/configuration selection is finished:
python test.py --data_root "$DATA_ROOT" --transpose_image --eval_split test --batch_size 1 --name public252_v1_baseline --pretrained_model_path exp/CRSDUN/public252_v1_baseline/model/best_iou.pth
```

Evaluation defaults to validation and requires explicit weights. Output folders distinguish validation/test, and prediction filenames preserve scene IDs. Evaluation writes metrics by default. Add --save_predictions only when full reconstructed cubes and segmentation images are needed. Record the checkpoint and protocol with any reported result. Never tune on the held-out test set.

## Local verification and remaining work

- CPU regression passed: exact next-step continuation including Adam/RNG/scheduler/scaler, incompatible resume rejection, split isolation, crop/flip alignment and progress reporting.
- CUDA smoke passed: one real training crop, one Adam update, full checkpoint reload and matching finite reconstruction. This is not a successful overfit experiment or a full-resolution validation memory test.
- Single-GPU training is the supported handoff. The upstream DDP path has alignment/split fixes and a rank-zero validation collective correction, but multi-GPU execution is untested and does not implement the new resumable checkpoint workflow.
- Complete the multiclass overfit diagnostic, full-resolution validation memory check and baseline training on the university GPU. Exact CRSDUN author dataset equivalence remains unresolved.
- Research plan, literature matrix, claims and experiment records are under `research/`. Large source PDFs and local diagnostic figures are not distributed in Git; primary URLs remain in the research documents.

No credentials belong in source files or Git. Keep dataset files and experiment outputs in their ignored directories; transfer artifacts separately when needed.

Kaggle-specific commands are in [KAGGLE_SETUP.md](KAGGLE_SETUP.md). Worker count now defaults to zero in the protocol launcher for portability; explicitly preserve the original worker count when resuming older checkpoints.

When moving a resumable run, transfer the **complete model folder** containing last.pt, best_iou.pth and best_psnr.pth together. The trainer preserves both prior best checkpoints in a new output directory, even if subsequent epochs do not improve.
