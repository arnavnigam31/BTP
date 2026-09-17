# Run public252 on Kaggle

The code is prepared for Kaggle, but has not been executed in a Kaggle session yet. Start with the checks below. Use one CUDA GPU; the new resumable workflow is single-GPU.

1. Create a GPU notebook. Internet is needed for cloning and installing packages; otherwise attach the code and required packages as inputs. GPU availability and session allowances depend on your account; inspect the notebook settings. [Kaggle GPU guidance](https://www.kaggle.com/page/GPU-tips-and-tricks) and [official notebook metadata options](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
2. Attach the existing public252 dataset as an input. Its folder must contain actual `visible_28/` and `labels/` files plus the original CSVs/manifests, not WSL symlinks. Choose the actual input directory from the notebook file panel. No upload or public dataset publication has been performed by this project.
3. Use `/kaggle/working` for code and output; leave attached inputs untouched. The examples below require substituting the attached dataset's actual path.

## Notebook cells

```python
!git clone --branch btp-public252-baseline https://github.com/arnavnigam31/BTP.git /kaggle/working/BTP
%cd /kaggle/working/BTP
```

Keep Kaggle's matched torch/torchvision GPU packages initially. Install only missing non-torch dependencies; the local pinned environment is in requirements-btp.txt, but blindly downgrading Kaggle's NumPy stack may break its preinstalled packages.

```python
%pip install einops torchmetrics==1.6.1
import torch, torchvision, numpy, pandas, scipy, skimage, PIL
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
assert torch.cuda.is_available(), "Select a GPU accelerator first"
print(torch.cuda.get_device_properties(0))
```

If installation changes packages that were already imported, restart the notebook session/kernel before continuing. Missing additional packages must be installed before the import check passes.

```python
from pathlib import Path
DATA_ROOT = Path('/kaggle/input/YOUR-DATASET/public252')  # EDIT THIS
assert (DATA_ROOT/'train_data.csv').is_file()
!python verify_evaluation.py
!python run_public252.py --root "{DATA_ROOT}"
!python verify_dataset_hashes.py --root "{DATA_ROOT}"
!python verify_portability.py --root "{DATA_ROOT}"
!python verify_public252_pipeline.py --root "{DATA_ROOT}"
!python gpu_smoke.py --root "{DATA_ROOT}" --out /kaggle/working/gpu_smoke
```

After transfer and infrastructure checks, complete the fixed-crop learning diagnostic before interpreting baseline performance:

```python
!python diagnostic_public252.py --root "{DATA_ROOT}" --out /kaggle/working/fixed_crop --transpose-image --steps 200
```

Start the intended baseline schedule (not a shortened schedule masquerading as the full run):

```python
!python run_public252.py --root "{DATA_ROOT}" --train --epochs 500 --batch-size 1 --workers 0 --output /kaggle/working/runs --name public252_v1_baseline
```

Save/export completed checkpoints before ending the session. A transient working directory is not durable checkpoint storage by itself. A later notebook may attach the previous saved output and resume from its `model/last.pt`:

```python
RESUME = '/kaggle/input/YOUR-SAVED-RUN/public252_v1_baseline/model/last.pt'  # EDIT
!python run_public252.py --root "{DATA_ROOT}" --train --epochs 500 --batch-size 1 --workers 0 --output /kaggle/working/runs --name public252_v1_baseline --resume "{RESUME}"
```

The total schedule, worker count, batch size and other checked settings must match the checkpoint. An interruption inside an epoch replays that epoch from its last completed checkpoint; it does not preserve partial-epoch updates.

## Validation and final testing

```python
WEIGHTS = '/kaggle/working/runs/public252_v1_baseline/model/best_iou.pth'
!python test.py --data_root "{DATA_ROOT}" --transpose_image --eval_split val --pretrained_model_path "{WEIGHTS}" --outf /kaggle/working/evaluations --name baseline_val
```

Use `--limit 1` with validation only for a full-resolution memory smoke check; the output is explicitly labeled `smoke_val`, not a full validation score. No test pixels need to be evaluated for this check. Full-frame evaluation might exceed the available GPU memory: no silent patching/downsampling fallback is used because that would change the protocol. Select adequate hardware if it fails.

Only after all model/configuration choices are frozen, replace `--eval_split val` with `--eval_split test` and choose a new run name. Evaluation requires a fresh output folder, records checkpoint/mask/protocol hashes and scene IDs, and writes aggregate/per-scene reconstruction metrics plus per-class segmentation metrics and a confusion matrix. `--save_predictions` additionally writes image cubes and colored segmentation maps; it is off by default to limit disk use.

See SERVER_SETUP.md for the separate-machine workflow and scientific limitations. Full baseline training, successful multiclass overfitting and target-platform full-frame validation remain pending.

When moving a resumable run, transfer the **complete model folder** containing last.pt, best_iou.pth and best_psnr.pth together. The trainer preserves both prior best checkpoints in a new output directory, even if subsequent epochs do not improve.
