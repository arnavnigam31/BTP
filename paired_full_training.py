"""Matched epoch51-60 LR experiment; never modifies the reference checkpoint."""
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


def branch_state(source, factor, initial_metrics):
    """Scale the entire remaining cosine LR schedule, retaining Adam moments/RNG."""
    if factor not in (1.0, 0.25):
        raise ValueError('This registered experiment uses only factors1 and0.25')
    state = copy.deepcopy(source)
    state['config']['learning_rate'] *= factor
    for group in state['optimizer']['param_groups']:
        group['lr'] *= factor
        if 'initial_lr' in group:
            group['initial_lr'] *= factor
    scheduler = state['scheduler']
    scheduler['base_lrs'] = [x*factor for x in scheduler['base_lrs']]
    scheduler['_last_lr'] = [x*factor for x in scheduler['_last_lr']]
    scheduler['eta_min'] *= factor
    # Select within this continuation, including its shared epoch50 starting point.
    # Do not inherit epoch49's better weights as a result of either arm.
    state['best'] = dict(initial_metrics)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True, help='Complete epoch50 run folder')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parent
    root, reference, out = a.root.resolve(), a.reference.resolve(), a.out.resolve()
    checkpoint = reference/'model/last.pt'
    source = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = dict(method='CRSDUN', batch_size=1, max_epoch=500, learning_rate=.0004,
                    transpose_image=True, input_setting='Y', input_mask='SSR', seed=3407, workers=0)
    if source.get('format_version') != 1 or source['epoch'] != 50:
        raise ValueError('Requires the epoch50 full training checkpoint')
    for key, value in expected.items():
        if source['config'].get(key) != value:
            raise ValueError(f'Unexpected reference configuration: {key}')
    if source['scheduler']['T_max'] != 500 or source['scheduler']['last_epoch'] != 50:
        raise ValueError('Unexpected scheduler state')
    actual_identity = identity(SimpleNamespace(data_root=str(root), mask_path=str(repo/'mask/mask512x512.mat')))
    if actual_identity != source['data_identity']:
        raise ValueError('Reference dataset metadata/mask identity does not match')
    initial = json.loads((reference/'result/epoch_0050_training.json').read_text())
    initial_metrics = {'iou': initial['val_foreground_miou_all22'], 'psnr': initial['val_psnr_ref1']}
    subprocess.run([sys.executable, str(repo/'run_public252.py'), '--root', str(root)], cwd=repo, check=True)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'reference_checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_epoch': 50, 'end_epoch': 60, 'epochs_per_arm': 10,
                'initial_metrics': initial_metrics, 'factors': {'control': 1.0, 'quarter_lr': .25},
                'best_selection_scope': 'Epoch50 initial weights and epochs51-60; historical epoch49 excluded',
                'change': 'Scale optimizer LR and remaining cosine schedule, including eta_min; retain moments, scaler, RNG and model',
                'limitations': 'Single seed, validation-only diagnostic; numerical hardware nondeterminism remains possible',
                'source_sha256': {f: hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in
                    ['paired_full_training.py', 'train.py', 'training_state.py', 'dataset.py', 'opt.py']}}
    (out/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name, factor in manifest['factors'].items():
        folder = out/'initial_states'/name
        folder.mkdir(parents=True)
        state = branch_state(source, factor, initial_metrics)
        atomic_save(state, folder/'last.pt')
        for filename in ('best_iou.pth', 'best_psnr.pth'):
            atomic_save(state['model'], folder/filename)
    if a.prepare_only:
        print('Prepared paired states; no GPU training launched.', flush=True)
        return
    summary = {'experiment': manifest, 'arms': {}}
    for name, factor in manifest['factors'].items():
        command = [sys.executable, '-u', 'train.py', '--data_root', str(root)+'/',
                   '--transpose_image', '--batch_size', '1', '--workers', '0', '--seed', '3407',
                   '--max_epoch', '500', '--stop_after_epoch', '60', '--learning_rate', str(.0004*factor),
                   '--resume', str(out/'initial_states'/name/'last.pt'), '--outf', str(out)+'/', '--name', name]
        subprocess.run(command, cwd=repo, check=True)
        records = [json.loads((out/name/'result'/f'epoch_{e:04d}_training.json').read_text()) for e in range(51,61)]
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
