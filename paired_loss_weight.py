"""Matched epoch101-110 segmentation-loss experiment; never modifies the reference checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch
from training_state import atomic_save, identity
from types import SimpleNamespace


def branch_state(source, weight, initial_metrics):
    """Change only recorded loss weight and branch-local best-selection bookkeeping."""
    if weight not in (1e-4, 1e-3):
        raise ValueError('Registered weights are1e-4 and1e-3')
    state = copy.deepcopy(source)
    state['config']['lambda_seg'] = weight
    state['best'] = dict(initial_metrics)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='Complete epoch100 run folder')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parent
    root, reference, out = a.root.resolve(), a.reference.resolve(), a.out.resolve()
    checkpoint = reference/'model/last.pt'
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = dict(method='CRSDUN', batch_size=1, max_epoch=500, learning_rate=.0001,
                    transpose_image=True, input_setting='Y', input_mask='SSR', seed=3407, workers=0)
    if source.get('format_version') != 1 or source['epoch'] != 100:
        raise ValueError('Requires the epoch100 full training checkpoint')
    for key, value in expected.items():
        if source['config'].get(key) != value:
            raise ValueError(f'Unexpected reference configuration: {key}')
    if source['scheduler']['T_max'] != 500 or source['scheduler']['last_epoch'] != 100:
        raise ValueError('Unexpected scheduler state')
    if source['config'].get('lambda_seg',1e-4)!=1e-4 or source['scheduler']['eta_min']!=2.5e-7:
        raise ValueError('Requires unchanged-loss quarter-rate reference checkpoint')
    actual_identity = identity(SimpleNamespace(data_root=str(root), mask_path=str(repo/'mask/mask512x512.mat')))
    if actual_identity != source['data_identity']:
        raise ValueError('Reference dataset metadata/mask identity does not match')
    initial = json.loads((reference/'result/epoch_0100_training.json').read_text())
    initial_metrics = {'iou': initial['val_foreground_miou_all22'], 'psnr': initial['val_psnr_ref1']}
    subprocess.run([sys.executable, str(repo/'run_public252.py'), '--root', str(root)], cwd=repo, check=True)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'reference_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_epoch': 100, 'end_epoch': 110, 'epochs_per_arm': 10,
                'initial_metrics': initial_metrics, 'weights': {'control_seg1e-4': 1e-4, 'higher_seg1e-3': 1e-3},
                'best_selection_scope': 'Epoch100 initial weights and epochs101-110; historical epoch96 excluded',
                'change': 'Change only lambda_seg; preserve quarter LR schedule, optimizer moments, scaler, RNG, model and sampling',
                'limitations': 'Single seed, validation-only diagnostic; numerical hardware nondeterminism remains possible',
                'source_sha256': {f: hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in
                    ['paired_loss_weight.py', 'train.py', 'training_state.py', 'dataset.py', 'opt.py']}}
    (out/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name, weight in manifest['weights'].items():
        folder = out/'initial_states'/name
        folder.mkdir(parents=True)
        state = branch_state(source, weight, initial_metrics)
        atomic_save(state, folder/'last.pt')
        for filename in ('best_iou.pth', 'best_psnr.pth'):
            atomic_save(state['model'], folder/filename)
    if a.prepare_only:
        print('Prepared paired states; no GPU training launched.', flush=True)
        return
    summary = {'experiment': manifest, 'arms': {}}
    for name, weight in manifest['weights'].items():
        command = [sys.executable, '-u', 'train.py', '--data_root', str(root)+'/',
                   '--transpose_image', '--batch_size', '1', '--workers', '0', '--seed', '3407',
                   '--max_epoch', '500', '--stop_after_epoch', '110', '--learning_rate', '0.0001', '--lambda_seg', str(weight),
                   '--resume', str(out/'initial_states'/name/'last.pt'), '--outf', str(out)+'/', '--name', name]
        subprocess.run(command, cwd=repo, check=True)
        records = [json.loads((out/name/'result'/f'epoch_{e:04d}_training.json').read_text()) for e in range(101,111)]
        miou = [r['val_foreground_miou_all22'] for r in records]
        summary['arms'][name] = {'final': records[-1], 'epochs': records,
            'best_logged_continuation_miou': max(miou), 'mean_last5_miou': statistics.mean(miou[-5:]),
            'std_last5_miou': statistics.pstdev(miou[-5:]),
            'actual_updates': sum(r['optimizer_updates'] for r in records),
            'amp_skips': sum(r['amp_skipped_updates'] for r in records)}
        (out/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
