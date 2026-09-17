"""Validate the frozen public252 protocol and optionally launch training.

Default is a read-only preflight. --train explicitly launches the baseline.
Keep public252-v1.json beside this file. Dataset contents are never edited.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, re

def validate(root, spec):
    for name, expected in spec['dataset_files_sha256'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Frozen dataset metadata changed: {name}. Create a new protocol version before training.')
    import pandas as pd
    import numpy as np
    from PIL import Image
    memberships={}
    for split, count in spec['split_sizes'].items():
        df=pd.read_csv(root/f'{split}_data.csv',index_col=0)
        ids=df.names.astype(str).tolist()
        if len(ids)!=count or len(set(ids))!=count or not df.masks.eq(True).all():
            raise ValueError(f'Invalid {split} membership')
        memberships[split]=set(ids)
        for name in ids:
            image=root/'visible_28'/f'{name}.npy'
            label=root/'labels'/f'{name}.png'
            x=np.load(image,mmap_mode='r',allow_pickle=False)
            if x.shape!=(512,512,28) or x.dtype!=np.float64:
                raise ValueError(f'Unexpected cube format: {image}')
            with Image.open(label) as y:
                if y.size!=(512,512) or y.mode!='RGB':
                    raise ValueError(f'Unexpected label format: {label}')
    if len(set.union(*memberships.values()))!=sum(map(len,memberships.values())):
        raise ValueError('Overlapping splits')
    return memberships

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('exp/CRSDUN'))
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--name',default='public252_v1_baseline')
    parser.add_argument('--stop-after-epoch',type=int,help='Absolute stopping epoch; does not alter the scheduler horizon')
    args=parser.parse_args()
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error('stop-after-epoch must be between 1 and epochs')
    if args.workers<0: parser.error('workers must be nonnegative')
    if args.resume and not args.train: parser.error('--resume requires --train')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.name):
        parser.error('Run name must contain only letters, digits, underscore, dot or hyphen')
    spec_path=Path(__file__).with_name('public252-v1.json')
    spec=json.loads(spec_path.read_text())
    if args.epochs<1 or args.batch_size<1: parser.error('Epochs and batch size must be positive')
    memberships=validate(args.root,spec)
    # Verify the actual loader preserves intensities and corrects spatial axes.
    sys.path.insert(0,str(args.repo))
    from dataset import HySpecSegmentation
    import numpy as np
    ds=HySpecSegmentation(str(args.root.resolve())+'/', 'train_data.csv',transpose_image=True)
    actual,label=ds.read_image_label(0)
    source=np.load(args.root/'visible_28'/f'{ds.names[0]}.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(actual,source.transpose(1,0,2)):
        raise ValueError('Loader violates the identity-intensity / transpose protocol')
    import torch
    from utils import Metrics_Rec
    from torchmetrics.functional.image import structural_similarity_index_measure
    target=torch.linspace(0,2,28*16*16).reshape(1,28,16,16)
    predicted=target+0.1
    metric=Metrics_Rec()
    metric.add_batch(target,predicted)
    expected_ssim=structural_similarity_index_measure(predicted,target,data_range=1.0).item()
    if abs(metric.psnr-20)>1e-4 or abs(metric.ssim-expected_ssim)>1e-6:
        raise ValueError('Reconstruction metric no longer matches fixed unit-reference protocol')
    command=[sys.executable,'train.py','--data_root',str(args.root.resolve())+'/',
             '--transpose_image','--batch_size',str(args.batch_size),'--max_epoch',str(args.epochs),
             '--name',args.name,'--workers',str(args.workers),'--outf',str(args.output.resolve())+'/']
    if args.stop_after_epoch is not None: command += ['--stop_after_epoch',str(args.stop_after_epoch)]
    if args.resume: command += ['--resume',str(args.resume.resolve())]
    print(json.dumps({'protocol':spec['protocol_id'],'protocol_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
          'preflight':'passed','splits':{k:len(v) for k,v in memberships.items()},
          'metrics':spec['reconstruction_metrics'],'training_command':command,
          'note':'Header/membership checks plus one training-scene loader check; full pixel audit is recorded separately.'},indent=2))
    if args.train:
        # Refuse accidental reuse of an existing experiment directory.
        run=args.output.resolve()/args.name
        if run.exists() and not args.resume: raise FileExistsError(f'Choose a new run name: {run}')
        run.mkdir(parents=True,exist_ok=bool(args.resume))
        (run/'dataset_protocol.json').write_bytes(spec_path.read_bytes())
        subprocess.run(command,cwd=args.repo,check=True)

if __name__=='__main__': main()
