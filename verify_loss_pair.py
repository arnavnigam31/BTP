"""CPU preservation, legacy resume and incompatible loss-setting checks."""
import copy,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from paired_loss_weight import branch_state
from training_state import save_state,load_state

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,500,2.5e-7)
scaler=torch.GradScaler('cpu')
optimizer.zero_grad();scaler.scale(model(torch.ones(1,2)).square().sum()).backward();scaler.step(optimizer);scaler.update();scheduler.step()
config=SimpleNamespace(method='CRSDUN',batch_size=1,max_epoch=500,learning_rate=.0001,transpose_image=True,input_setting='Y',input_mask='SSR',seed=3407,workers=0)
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'legacy.pt';save_state(path,model,optimizer,scheduler,scaler,100,{'iou':.53,'psnr':35.85},config,{'fixture':'same'})
 source=torch.load(path,map_location='cpu',weights_only=True);before=copy.deepcopy(source)
 for weight in [1e-4,1e-3]:
  state=branch_state(source,weight,source['best'])
  for key in ['model','optimizer','scheduler','rng','scaler','data_identity','epoch']:
   assert equal(source[key],state[key]),key
  branch=Path(tmp)/f'{weight}.pt';torch.save(state,branch)
  config.lambda_seg=weight
  assert load_state(branch,model,optimizer,scheduler,scaler,config,{'fixture':'same'})[0]==100
  config.lambda_seg=1e-3 if weight==1e-4 else 1e-4
  try:load_state(branch,model,optimizer,scheduler,scaler,config,{'fixture':'same'})
  except ValueError as exc:assert 'lambda_seg' in str(exc)
  else:raise AssertionError('Mismatched loss silently accepted')
 config.lambda_seg=1e-4
 load_state(path,model,optimizer,scheduler,scaler,config,{'fixture':'same'})
 assert equal(before,source)
 # This is the unchanged weighted objective; larger lambda scales segmentation gradients only.
 grads=[]
 for weight in [1e-4,1e-3]:
  logits=torch.tensor([[.2,-.3]],requires_grad=True)
  (weight*torch.nn.functional.cross_entropy(logits,torch.tensor([1]))).backward()
  grads.append(logits.grad.clone())
 torch.testing.assert_close(grads[1],10*grads[0])
print('PASS: both loss branches retain optimizer/scheduler/model/RNG/scaler; legacy resume accepted; incompatible loss rejected; segmentation gradient scaling verified.')
