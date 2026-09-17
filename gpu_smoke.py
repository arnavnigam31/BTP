"""One training crop: CUDA optimizer/checkpoint/reload smoke test, not an overfit experiment."""
import argparse,gc,json
from pathlib import Path
from types import SimpleNamespace
import torch
from dataset import HySpecSegmentation,encode_segmap
from models import model_generator
from utils import init_mask,crop_mask,init_meas,set_seed
from training_state import save_state,load_state,identity
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);set_seed(3407)
opt=SimpleNamespace(data_root=a.root,mask_path='mask/mask512x512.mat',method='CRSDUN',batch_size=1,
                    max_epoch=2,learning_rate=.0004,transpose_image=True,input_setting='Y',input_mask='SSR',seed=3407,workers=0)
ds=HySpecSegmentation(a.root,'train_data.csv',transpose_image=True)
x,y=ds.read_image_label(0)
x=torch.from_numpy(x[170:426,128:384].copy()).permute(2,0,1).unsqueeze(0).float().cuda()
y=torch.from_numpy(encode_segmap(y)[170:426,128:384].copy()).unsqueeze(0).long().cuda()
phi,mask=init_mask(opt.mask_path,'SSR',1,28)
phi,mask=crop_mask(phi,mask,opt,28);measurement=init_meas(x,phi,'Y')
def create():
    model=model_generator('CRSDUN',28,23)
    optim=torch.optim.Adam(model.parameters(),lr=.0004)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,2)
    scaler=torch.GradScaler('cuda',init_scale=1024)
    return model,optim,scheduler,scaler
model,optim,scheduler,scaler=create();model.train()
with torch.autocast('cuda'):
    rec,seg=model(measurement,mask)
    loss=sum((.7**i)*((r-x).square().mean()+.0001*torch.nn.functional.cross_entropy(s,y)) for i,(r,s) in enumerate(zip(rec[::-1],seg[::-1])))
scaler.scale(loss).backward();scaler.step(optim);scaler.update();scheduler.step()
assert optim.state and all(int(v['step'])==1 for v in optim.state.values())
loss_value=float(loss.detach());del rec,seg,loss
model.eval()
with torch.no_grad():before=model(measurement,mask)[0][-1].cpu()
save_state(out/'last.pt',model,optim,scheduler,scaler,1,{'iou':0.,'psnr':0.},opt,identity(opt))
del model,optim,scheduler,scaler;gc.collect();torch.cuda.empty_cache()
model,optim,scheduler,scaler=create()
epoch,_=load_state(out/'last.pt',model,optim,scheduler,scaler,opt,identity(opt))
assert epoch==1 and all(int(v['step'])==1 for v in optim.state.values())
model.eval()
with torch.no_grad():after=model(measurement,mask)[0][-1].cpu()
assert torch.isfinite(after).all()
torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
result={'status':'passed','scene':str(ds.names[0]),'loss':loss_value,'optimizer_updates':1,
        'checkpoint_reload':True,'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'scope':'One crop; not a passed overfit test or full-resolution validation memory benchmark'}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
