# Paired learning-rate diagnostic

Purpose: investigate unstable six-class training-crop predictions observed at step 1,000. This is an optimization diagnostic, not a journal contribution, full baseline, or held-out evaluation.

Run two sequential arms from the SAME saved 1,000-step checkpoint:

| Arm | Learning rate | Additional attempted updates |
| --- | --- | --- |
| Control | 0.0004 | 500 |
| Lower rate | 0.00004 | 500 |

Both arms restore identical Adam moments and AMP scaler state. Crops, measurement mask, crop alternation, loss weights and precision settings remain fixed. No scheduler is introduced. The legacy checkpoint lacks RNG state, so both arms start with the same fresh seed 3407; this is a matched new continuation, not bitwise continuation of the old session. New checkpoints save RNG. The original mask crop is reconstructed with the original seed/code; the source images and labels are hashed and initial metrics must agree with the saved final metrics before proceeding.

Compare final metrics, the mean and standard deviation of the last ten logged mIoUs, all six per-class IoUs, and actual/skipped optimizer updates. A transient maximum alone is insufficient evidence of stability. This single paired run cannot establish broad superiority or generalization. Do not change labels/architecture/losses during this comparison.

## Offline Kaggle setup

Attach these inputs to a GPU notebook:

1. Existing `btp-data` prepared dataset.
2. Updated code archive `BTP-code-lr-comparison.zip`, uploaded privately.
3. Saved `fixed_crop_1000_backup.zip` containing the complete 1,000-step output directory, uploaded privately or attached as a saved notebook output.

Do not delete or overwrite the old run. If Kaggle leaves ZIPs compressed, this cell extracts only the recognized code/run archives into working storage. It can also use already extracted inputs. It refuses ambiguous matches.

```python
from pathlib import Path
import shutil, zipfile, os

inputs = Path('/kaggle/input')
working = Path('/kaggle/working')

def safe_extract(archive, target):
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (target / name).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe ZIP member')
        z.extractall(target)

code = list(inputs.rglob('compare_diagnostic_lr.py'))
if not code:
    for archive in inputs.rglob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            is_code = 'BTP/compare_diagnostic_lr.py' in z.namelist()
        if is_code:
            target = working / 'lr_code_unpacked'
            safe_extract(archive, target)
            code.extend(target.rglob('compare_diagnostic_lr.py'))
assert len(code) == 1, f'Expected one updated code input: {code}'
shutil.copytree(code[0].parent, working/'BTP', dirs_exist_ok=True)
os.chdir(working/'BTP')

run_name = 'fixed_crop_1000_20260917_145850'
runs = [p.parent for p in inputs.rglob('diagnostic_checkpoint.pt')
        if p.parent.name == run_name]
if not runs:
    for archive in inputs.rglob('*.zip'):
        with zipfile.ZipFile(archive) as z:
            is_run = f'{run_name}/diagnostic_checkpoint.pt' in z.namelist()
        if is_run:
            target = working / 'reference_unpacked'
            safe_extract(archive, target)
            runs.append(target/run_name)
assert len(runs) == 1, f'Expected one saved 1,000-step run: {runs}'
REFERENCE_RUN = runs[0]
DATA_ROOT = Path('/kaggle/input/datasets/arnavnigamd/btp-data/public252')
assert (DATA_ROOT/'train_data.csv').is_file()
print('Code ready; reference:', REFERENCE_RUN)
```

This cell is intended to run once in a fresh session; if already extracted, reuse its variables rather than extracting again. Check installed packages as before. Then run:

```python
import subprocess, sys, torch
from datetime import datetime
assert torch.cuda.is_available(), 'Enable a CUDA GPU first'
subprocess.run([sys.executable, 'verify_diagnostic_comparison.py'], check=True)
OUT = Path('/kaggle/working/lr_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
subprocess.run([
    sys.executable, 'compare_diagnostic_lr.py',
    '--root', str(DATA_ROOT), '--from-run', str(REFERENCE_RUN),
    '--out', str(OUT), '--steps', '500', '--eval-every', '10'
], check=True)
print('Results:', OUT)
print((OUT/'comparison_summary.json').read_text())
```

Expect roughly 20–30 minutes based on the earlier Kaggle diagnostic speed; checkpoint writes and session load can change this. The script uses one GPU. Each arm saves latest and best full checkpoints and matching prediction images, with evaluation/backup every ten attempts. Best selection is on these training crops only. `comparison_summary.json` contains both arms; if execution is interrupted after one arm, a partial summary is not a completed comparison. Preserve the complete output folder before ending the session.
