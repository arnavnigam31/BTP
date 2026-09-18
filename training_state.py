"""Epoch-boundary checkpoints, dataset identity and run provenance."""
from pathlib import Path
import hashlib, json, os, random, subprocess, sys
import numpy as np
import torch

def identity(opt):
    root=Path(opt.data_root)
    files=list(root.glob('*_data.csv'))+list(root.glob('*manifest.json'))
    result={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    result['mask']=hashlib.sha256(Path(opt.mask_path).read_bytes()).hexdigest()
    return result

def rng_state():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])

def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);os.replace(temporary,path)

def save_state(path,model,optimizer,scheduler,scaler,epoch,best,opt,data_id):
    atomic_save({'format_version':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
                 'best':best,'config':vars(opt),'data_identity':data_id,'rng':rng_state()},path)

def load_state(path,model,optimizer,scheduler,scaler,opt,data_id):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('format_version')!=1:raise ValueError('Resume requires a full training checkpoint')
    if state['data_identity']!=data_id:raise ValueError('Dataset metadata or measurement mask changed')
    for key in ['method','batch_size','max_epoch','learning_rate','transpose_image','input_setting','input_mask','seed','workers']:
        if state['config'].get(key)!=getattr(opt,key):raise ValueError(f'Resume setting differs: {key}')
    if state['config'].get('lambda_seg',1e-4) != getattr(opt,'lambda_seg',1e-4):
        raise ValueError('Resume setting differs: lambda_seg')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state['epoch'],state['best']

def record_run(path,opt,data_id):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    def git(*args):
        p=subprocess.run(['git',*args],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else 'unavailable'
    suffix='resume' if opt.resume else 'start'
    record={'config':vars(opt),'data_identity':data_id,'git_commit':git('rev-parse','HEAD'),
            'git_status':git('status','--short'),'python':sys.version,'torch':str(torch.__version__),
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'checkpoint_policy':'Epoch boundary; last.pt resumable; best_iou.pth and best_psnr.pth weights only',
            'metrics':'PSNR/SSIM reference 1; foreground mIoU over all 22 IDs, absent IDs score zero (upstream epsilon convention).'}
    (path/f'run_{suffix}.json').write_text(json.dumps(record,indent=2)+'\n')
    (path/f'source_{suffix}.diff').write_text(git('diff','HEAD')+'\n')


def carry_best_weights(resume_path, destination):
    """Preserve previously selected weights when continuing in a new run folder."""
    import shutil, filecmp
    source=Path(resume_path).resolve().parent
    destination=Path(destination).resolve()
    if source==destination:return
    pairs=[(source/name,destination/name) for name in ('best_iou.pth','best_psnr.pth')]
    for old,new in pairs:
        if not old.is_file():raise FileNotFoundError(f'Transfer the complete model folder; missing {old}')
        if new.exists() and not filecmp.cmp(old,new,shallow=False):
            raise FileExistsError(f'Refusing to replace different selected weights: {new}')
    destination.mkdir(parents=True,exist_ok=True)
    for old,new in pairs:
        if not new.exists():shutil.copyfile(old,new)
