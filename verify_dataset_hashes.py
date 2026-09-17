"""Verify every transferred candidate cube against the audited array hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import get_labels
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
rows=json.loads((Path(__file__).parent/'research/dataset_audit_20260916/scenes.json').read_text())
label_hashes=json.loads((Path(__file__).parent/'research/dataset_audit_20260916/label_file_sha256.json').read_text())
palette={tuple(x) for x in get_labels().tolist()}
for i,r in enumerate(rows):
    image=np.load(a.root/'visible_28'/(r['scene']+'.npy'),allow_pickle=False)
    if list(image.shape)!=r['shape28'] or str(image.dtype)!=r['dtype']:
        raise ValueError('Unexpected cube format: '+r['scene'])
    if hashlib.sha256(image.tobytes()).hexdigest()!=r['sha256_cube28']:
        raise ValueError('Cube differs from audited content: '+r['scene'])
    label_path=a.root/'labels'/(r['scene']+'.png')
    if hashlib.sha256(label_path.read_bytes()).hexdigest()!=label_hashes[r['scene']]:
        raise ValueError('Label differs from audited content: '+r['scene'])
    with Image.open(label_path) as f:label=np.array(f)
    if label.shape!=(512,512,3) or not set(map(tuple,np.unique(label.reshape(-1,3),axis=0))).issubset(palette):
        raise ValueError('Invalid label shape/palette: '+r['scene'])
    if (i+1)%25==0:print(f'Checked {i+1}/{len(rows)}',flush=True)
print('PASS: all candidate cube array hashes, label file hashes and label formats/palettes')
