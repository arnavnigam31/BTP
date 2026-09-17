"""CPU checks for public252 alignment and training progress (dataset stays read-only)."""
import argparse
import ast
from pathlib import Path
import numpy as np
from dataset import (HySpecSegmentation, prep_loaders, prep_loaders_ddp,
                     RandomCropHoriz, SegIdentityTransform, RandomHorizontalFlip)

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
args = p.parse_args()
root = str(Path(args.root))+'/'
original = HySpecSegmentation(root, 'train_data.csv')
aligned = HySpecSegmentation(root, 'train_data.csv', transpose_image=True)
x, y = original.read_image_label(0)
xx, yy = aligned.read_image_label(0)
assert np.array_equal(xx, x.transpose(1, 0, 2))
assert np.array_equal(y, yy)
for fn, kw in [(prep_loaders, {}), (prep_loaders_ddp, {'rank': 0, 'world_size': 1})]:
    tr, va = fn(root, workers=0, transpose_image=True, **kw)
    assert tr.dataset.transpose_image and va.dataset.transpose_image
    assert set(va.dataset.names) == set(HySpecSegmentation(root, 'val_data.csv').names)
    batch = next(iter(tr))
    assert batch['image'].shape == (1, 28, 256, 256)
    assert batch['label'].shape == (1, 256, 256)

# Coordinate-coded inputs expose crop/flip registration errors.
y = np.arange(512*512).reshape(512, 512)
x = np.repeat(y[..., None], 28, axis=2)
z = RandomHorizontalFlip(prob=1)(SegIdentityTransform()(RandomCropHoriz()({'image': x, 'label': y})))
assert np.array_equal(z['image'][0].numpy(), z['label'].numpy())

# Evaluate the production logging condition without allocating a GPU model.
tree = ast.parse(Path(__file__).with_name('train.py').read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
epoch = next(n for n in main.body if isinstance(n, ast.For))
batch = next(n for n in epoch.body if isinstance(n, ast.For))
assert batch.target.elts[0].id == 'batch_idx'
for nested in ast.walk(batch):
    if isinstance(nested, ast.For) and nested is not batch:
        assert not any(isinstance(n, ast.Name) and n.id == 'batch_idx' for n in ast.walk(nested.target))
progress = next(n for n in batch.body if isinstance(n, ast.If))
assert not any(isinstance(n, ast.Name) and n.id == 'i' for n in ast.walk(progress))
condition = compile(ast.Expression(progress.test), '<progress>', 'eval')
for length in (1, 10, 11, 202):
    actual = [i+1 for i in range(length) if eval(condition, {'batch_idx': i, 'train_loader': range(length)})]
    expected = sorted(set(list(range(10, length+1, 10))+[length]))
    assert actual == expected, (actual, expected)
print('PASS: alignment opt-in, unchanged labels, single/DDP loaders, crop/flip registration, batch progress')
