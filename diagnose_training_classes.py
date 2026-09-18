"""Training-only inference diagnostic; fixed class-rich crops, no optimizer updates."""
import argparse,json,hashlib,random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
from dataset import encode_segmap,get_class_names
from test import evaluate_batches,sha256

CLASSES=[9,12,17,22]

def choose(root):
    candidates={k:[] for k in CLASSES}
    for name in sorted(pd.read_csv(root/'train_data.csv',index_col=0).names.astype(str)):
        y=encode_segmap(np.asarray(Image.open(root/'labels'/f'{name}.png')))
        assert y.shape==(512,512)
        for k in CLASSES:
            full=int((y==k).sum())
            if not full:continue
            sums=(y[170:426]==k).sum(0);prefix=np.r_[0,np.cumsum(sums)]
            cols=np.arange(10,246);counts=prefix[cols+256]-prefix[cols]
            # Best legal coverage; ties resolve closest to centered crop, then smaller col.
            col=min(cols[counts==counts.max()],key=lambda c:(abs(int(c)-128),int(c)))
            candidates[k].append({'class_id':k,'scene':name,'row':170,'col':int(col),
                                  'class_pixels_in_crop':int(counts.max()),'class_pixels_full':full})
    return [v for k in CLASSES for v in sorted(candidates[k],key=lambda v:(-v['class_pixels_in_crop'],v['scene']))[:2]]

def slice_ssr(phi,mask,row,col):
    return (phi[:,:,row:row+256,col:col+310],
            (mask[0][:,:,row:row+256,col:col+256],mask[1][:,:,row:row+256,col:col+310],mask[2][:,:,row:row+256,col:col+310]))

class OneScene(Dataset):
    num_classes=23;class_names=get_class_names()
    def __init__(self,name,x,y):self.names=[name];self.x=x;self.y=y
    def __len__(self):return 1
    def __getitem__(self,i):return {'image':self.x,'label':self.y}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parent
    from run_public252 import validate
    validate(a.root,json.loads((repo/'public252-v1.json').read_text()))
    selection=choose(a.root)
    assert len(selection)==8,'Need two eligible training scenes per target class'
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2))
    record={'scope':'Selected training scenes only; deliberately class-rich, not an unbiased train metric or validation benchmark',
            'selection':'Two training scenes per target class with most class pixels in a legal crop; chosen without model predictions',
            'modes':['full','full_roi','crop_registered','crop_training_mask'],
            'comparison':'full_roi and both crop modes score identical image/label pixels. Crop remeasurement changes boundary/context; registered and training-mask modes also distinguish mask location.',
            'model_mode':'eval, float32, no augmentation or optimization',
            'protocol_sha256':sha256(repo/'public252-v1.json'),'test_pixels_read':False}
    if a.prepare_only:
        (a.out/'diagnostic.json').write_text(json.dumps(record,indent=2));print(json.dumps(selection,indent=2));return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if sha256(a.checkpoint)!='2d125c99ea2c2f15ef67240ba806514754218f57a2255b025e16bb2b5c969c03':
        raise ValueError('Use the audited epoch96 best_iou.pth from the epoch100 archive')
    from utils import init_mask,init_meas,set_seed
    from models import model_generator
    set_seed(3407)
    model=model_generator('CRSDUN',28,23,str(a.checkpoint));model.eval()
    phi,mask=init_mask(str(repo/'mask/mask512x512.mat'),'SSR',1,28)
    record.update(checkpoint_sha256=sha256(a.checkpoint),mask_sha256=sha256(repo/'mask/mask512x512.mat'),
                  status='running',gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__))
    (a.out/'diagnostic.json').write_text(json.dumps(record,indent=2))
    summaries=[]
    for i,sel in enumerate(selection):
        scene=sel['scene'];row,col=sel['row'],sel['col']
        x=np.load(a.root/'visible_28'/f'{scene}.npy').transpose(1,0,2).astype(np.float32)
        y=encode_segmap(np.asarray(Image.open(a.root/'labels'/f'{scene}.png')))
        x=torch.from_numpy(x.copy()).permute(2,0,1);y=torch.from_numpy(y.astype(np.int64))
        cropx=x[:,row:row+256,col:col+256];cropy=y[row:row+256,col:col+256]
        prefix=a.out/f'class{sel["class_id"]}_{scene}'
        cache={}
        class Capture(torch.nn.Module):
            def forward(self,mea,m):
                xr,yr=model(mea,m)
                cache['x']=xr[-1].detach().cpu();cache['y']=yr[-1].detach().cpu()
                return xr,yr
        class CachedROI(torch.nn.Module):
            def forward(self,mea,m):
                return [cache['x'][:,:,row:row+256,col:col+256]],[cache['y'][:,:,row:row+256,col:col+256]]
        rng=random.Random(3407+i);mr,mc=rng.randint(0,255),rng.randint(0,255)
        registered_phi,registered_mask=slice_ssr(phi,mask,row,col)
        training_phi,training_mask=slice_ssr(phi,mask,mr,mc)
        modes=[('full',Capture(),x,y,phi,mask,'cuda'),
               ('full_roi',CachedROI(),cropx,cropy,None,None,'cpu'),
               ('crop_registered',model,cropx,cropy,registered_phi,registered_mask,'cuda'),
               ('crop_training_mask',model,cropx,cropy,training_phi,training_mask,'cuda')]
        for mode,network,xx,yy,pp,mm,device in modes:
            folder=prefix/mode;folder.mkdir(parents=True)
            measure=(lambda t:t) if mode=='full_roi' else (lambda t,pp=pp:init_meas(t,pp,'Y'))
            evaluate_batches(network,DataLoader(OneScene(scene,xx,yy)),measure,mm,device,folder,save_segmentation=True)
            hist=np.load(folder/'confusion_matrix.npy');k=sel['class_id'];den=hist[k].sum();union=den+hist[:,k].sum()-hist[k,k]
            summaries.append(dict(**sel,mode=mode,mask_row=mr if mode=='crop_training_mask' else (row if mode=='crop_registered' else 0),
                mask_col=mc if mode=='crop_training_mask' else (col if mode=='crop_registered' else 0),
                true_pixels=int(den),target_recall=float(hist[k,k]/den) if den else None,
                target_iou=float(hist[k,k]/union) if union else None,
                top_predictions=[{'id':int(j),'fraction':float(hist[k,j]/den)} for j in np.argsort(hist[k])[::-1][:5]]))
        (a.out/'summary.json').write_text(json.dumps(summaries,indent=2))
        print(f'Completed training diagnostic {i+1}/{len(selection)}',flush=True)
    record['status']='completed';record['records']=len(summaries)
    (a.out/'diagnostic.json').write_text(json.dumps(record,indent=2))

if __name__=='__main__':main()
