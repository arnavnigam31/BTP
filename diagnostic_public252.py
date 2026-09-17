"""Read-only data audit and fixed-crop CRSDUN learning diagnostic."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from dataset import get_labels, HySpecSegmentation
from models import model_generator
from utils import set_seed, init_mask, crop_mask, init_meas

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=60)
p.add_argument('--audit-only', action='store_true')
p.add_argument('--transpose-image', action='store_true')
a = p.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
set_seed(3407)
torch.set_num_threads(4)
palette = get_labels().astype(np.uint32)
keys = palette[:, 0]*65536 + palette[:, 1]*256 + palette[:, 2]
order = np.argsort(keys)

def read_label(path):
    rgb = np.asarray(Image.open(path)).astype(np.uint32)
    assert rgb.ndim == 3 and rgb.shape[2] == 3, (path, rgb.shape)
    packed = rgb[..., 0]*65536 + rgb[..., 1]*256 + rgb[..., 2]
    pos = np.searchsorted(keys[order], packed)
    assert np.all(pos < len(keys)), ('Unknown label color', path)
    assert np.all(keys[order][pos] == packed), ('Unknown label color', path)
    return order[pos].astype(np.int64)

splits = {s: pd.read_csv(root/f'{s}_data.csv', index_col=0) for s in ('train', 'val', 'test')}
names = {s: list(df.names.astype(str)) for s, df in splits.items()}
for s, ns in names.items():
    assert len(ns) == len(set(ns)), ('Duplicate scene', s)
    assert splits[s].masks.all(), ('Disabled label', s)
    for n in ns:
        assert (root/'visible_28'/f'{n}.npy').is_file()
        assert (root/'labels'/f'{n}.png').is_file()
for s, t in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    assert not set(names[s]) & set(names[t]), ('Split overlap', s, t)
assert set(sum(names.values(), [])) == set(pd.read_csv(root/'all_data.csv', index_col=0).names.astype(str))
audit = {'split_sizes': {s: len(ns) for s, ns in names.items()},
         'scene_id_overlap': False, 'test_pixels_read': False,
         'note': 'Scene IDs checked; related-scene/group leakage not established by this audit.', 'splits': {}}
candidates = []
rng = np.random.default_rng(3407)
for s in ('train', 'val'):
    hist = np.zeros(23, dtype=np.int64)
    crop_fg = []
    lo, hi = float('inf'), float('-inf')
    for idx, n in enumerate(names[s]):
        x = np.load(root/'visible_28'/f'{n}.npy', mmap_mode='r')
        y = read_label(root/'labels'/f'{n}.png')
        assert x.shape == (*y.shape, 28), (n, x.shape, y.shape)
        assert np.isfinite(x).all(), ('Nonfinite image', n)
        lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
        hist += np.bincount(y.ravel(), minlength=23)
        if s == 'train':
            assert y.shape[0] >= 426 and y.shape[1] > 276
            for col in rng.integers(10, y.shape[1]-266, size=8):
                fg = float((y[170:426, col:col+256] > 0).mean())
                crop_fg.append(fg)
                candidates.append((fg, n, int(col)))
        if (idx+1) % 25 == 0:
            print(f'Audit {s}: {idx+1}/{len(names[s])}', flush=True)
    info = {'pixel_counts': hist.tolist(), 'missing_classes': np.flatnonzero(hist == 0).tolist(),
            'foreground_fraction': float(hist[1:].sum()/hist.sum()), 'image_min': lo, 'image_max': hi}
    if crop_fg:
        info['sampled_crop_foreground_quantiles'] = np.quantile(crop_fg, [0, .25, .5, .75, 1]).tolist()
        info['empty_crop_fraction'] = float(np.mean(np.array(crop_fg) == 0))
    audit['splits'][s] = info
(out/'audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
selected = []
for fg, n, col in sorted(candidates, reverse=True):
    if n not in [v['scene'] for v in selected]:
        selected.append({'scene': n, 'row': 170, 'col': col, 'foreground_fraction': fg})
    if len(selected) == 2:
        break
(out/'selection.json').write_text(json.dumps(selected, indent=2))
xs, ys = [], []
loader = HySpecSegmentation(str(root)+'/', 'train_data.csv', transpose_image=a.transpose_image)
for sel in selected:
    n, c = sel['scene'], sel['col']
    aligned_image, _ = loader.read_image_label(list(loader.names).index(n))
    xs.append(aligned_image[170:426, c:c+256].astype(np.float32))
    ys.append(read_label(root/'labels'/f'{n}.png')[170:426, c:c+256].copy())
def preview(preds=None, filename='crops.png'):
    canvas = Image.new('RGB', (3*276, 2*300), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, (x, y) in enumerate(zip(xs, ys)):
        rgb = x[..., [24, 18, 8]]
        rgb = (np.clip(rgb/max(float(np.quantile(rgb, .99)), 1e-8), 0, 1)*255).astype(np.uint8)
        colors = get_labels()[y].astype(np.uint8)
        overlay = rgb.copy()
        overlay[y > 0] = (.45*rgb[y > 0]+.55*colors[y > 0]).astype(np.uint8)
        last = get_labels()[y if preds is None else preds[j]].astype(np.uint8)
        for k, (im, label) in enumerate([(rgb, selected[j]['scene']+' pseudo-RGB'), (overlay, 'Ground-truth overlay'), (last, 'Ground truth' if preds is None else 'Prediction')]):
            canvas.paste(Image.fromarray(im), (k*276+10, j*300+30))
            draw.text((k*276+10, j*300+10), label, fill='black')
    canvas.save(out/filename)
preview()
if a.audit_only:
    raise SystemExit()
config = {'transpose_image': a.transpose_image, 'steps': a.steps, 'seed': 3407, 'learning_rate': .0004, 'lambda_rec': 1,
          'lambda_seg': .0001, 'fixed_training_crops': True, 'fixed_mask': True,
          'scheduler': None, 'initialization': 'scratch', 'selection': 'two highest-foreground distinct training scenes',
          'purpose': 'memorization diagnostic, not validation or test performance'}
(out/'config.json').write_text(json.dumps(config, indent=2))
device_x = [torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda() for x in xs]
device_y = [torch.from_numpy(y).unsqueeze(0).cuda() for y in ys]
phi, masks = init_mask('mask/mask512x512.mat', 'SSR', 1, 28)
phi, masks = crop_mask(phi, masks, SimpleNamespace(input_mask='SSR'), 28)
measurements = [init_meas(x, phi, 'Y') for x in device_x]
model = model_generator('CRSDUN', 28, 23)
optimizer = torch.optim.Adam(model.parameters(), lr=.0004)
scaler = torch.GradScaler('cuda')
start = time.time()
records = []
def evaluate(step):
    model.eval()
    hist = np.zeros((23, 23), dtype=np.int64)
    ce, mse, preds = [], [], []
    with torch.no_grad():
        for x, y, mea in zip(device_x, device_y, measurements):
            with torch.autocast(device_type='cuda'):
                xr, yr = model(mea, masks)
                ce.append(float(torch.nn.functional.cross_entropy(yr[-1], y)))
                mse.append(float(torch.nn.functional.mse_loss(xr[-1], x)))
            pred = yr[-1].argmax(1).cpu().numpy()[0]
            preds.append(pred)
            hist += np.bincount((23*y.cpu().numpy()[0]+pred).ravel(), minlength=529).reshape(23,23)
    union = hist.sum(0)+hist.sum(1)-hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.zeros(23), where=union>0)
    present = hist.sum(1)>0; present[0] = False
    record = {'step': step, 'elapsed_seconds': time.time()-start, 'ce': float(np.mean(ce)),
              'mse': float(np.mean(mse)), 'foreground_miou_present': float(iou[present].mean()),
              'foreground_recall': float(hist.diagonal()[1:].sum()/hist[1:].sum()),
              'predicted_foreground_fraction': float(hist[:,1:].sum()/hist.sum()),
              'prediction_counts': hist.sum(0).tolist(), 'iou_by_class': iou.tolist()}
    records.append(record)
    (out/'metrics.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    model.train()
    return preds
evaluate(0)
for step in range(1, a.steps+1):
    j = (step-1)%2
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda'):
        xr, yr = model(measurements[j], masks)
        rec = sum(torch.nn.functional.mse_loss(xr[k], device_x[j])*.7**i for i,k in enumerate(reversed(range(len(xr)))))
        seg = sum(torch.nn.functional.cross_entropy(yr[k], device_y[j])*.7**i for i,k in enumerate(reversed(range(len(yr)))))
        loss = rec + .0001*seg
    assert torch.isfinite(loss), 'Nonfinite loss'
    scaler.scale(loss).backward()
    if step == 1:
        grads = {n: float(v.grad.detach().norm()) for n,v in model.named_parameters() if v.grad is not None and (n.startswith('net_stage.') and int(n.split('.')[1]) % 3 == 2)}
        (out/'initial_scaled_gradient_norms.json').write_text(json.dumps(grads, indent=2))
    scaler.step(optimizer); scaler.update()
    print(f'Completed step {step}/{a.steps}; elapsed {time.time()-start:.1f}s', flush=True)
    if step % 10 == 0 or step == a.steps:
        preds = evaluate(step)
preview(preds, 'predictions.png')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'step': a.steps, 'config': config}, out/'diagnostic_checkpoint.pt')
print('Diagnostic finished', flush=True)
