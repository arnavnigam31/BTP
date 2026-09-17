"""CPU regression checks for resume, split selection and frozen metadata."""
import argparse, copy, json, random, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from training_state import save_state, load_state
from dataset import evaluation_loader
from run_public252 import validate

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
torch.set_num_threads(2)
root=Path(a.root)
spec=json.loads(Path(__file__).with_name('public252-v1.json').read_text())
validate(root,spec)
val=evaluation_loader(root,'val');test=evaluation_loader(root,'test')
assert len(val.dataset)==25 and len(test.dataset)==25
assert not set(val.dataset.names)&set(test.dataset.names)
try:evaluation_loader(root,'train')
except ValueError:pass
else:raise AssertionError('Invalid evaluation split accepted')
opt=SimpleNamespace(method='linear_test',batch_size=2,max_epoch=5,learning_rate=.001,
                    transpose_image=True,input_setting='Y',input_mask='SSR',seed=17,workers=0)
torch.manual_seed(17);np.random.seed(17);random.seed(17)
def create():
    m=torch.nn.Linear(3,2);o=torch.optim.Adam(m.parameters(),lr=.001)
    s=torch.optim.lr_scheduler.CosineAnnealingLR(o,5)
    g=torch.amp.GradScaler('cpu')
    return m,o,s,g
def step(m,o,s,g):
    x=torch.randn(2,3);y=torch.randn(2,2)
    o.zero_grad();loss=(m(x)-y).square().mean();g.scale(loss).backward();g.step(o);g.update();s.step()
    return [random.random(),float(np.random.random())]
m,o,s,g=create();step(m,o,s,g)
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'last.pt'
    save_state(path,m,o,s,g,1,{'iou':0.1,'psnr':20.0},opt,{'fixture':'fixed'})
    reference_rng=step(m,o,s,g);reference=copy.deepcopy(m.state_dict())
    m2,o2,s2,g2=create()
    epoch,best=load_state(path,m2,o2,s2,g2,opt,{'fixture':'fixed'})
    assert epoch==1 and best['iou']==.1
    assert step(m2,o2,s2,g2)==reference_rng
    for k,v in reference.items():torch.testing.assert_close(m2.state_dict()[k],v,rtol=0,atol=0)
    assert s.get_last_lr()==s2.get_last_lr() and g.state_dict()==g2.state_dict()
    for changed_opt,changed_id in [(opt,{'fixture':'changed'}),(SimpleNamespace(**{**vars(opt),'max_epoch':6}),{'fixture':'fixed'})]:
        try:load_state(path,m2,o2,s2,g2,changed_opt,changed_id)
        except ValueError:pass
        else:raise AssertionError('Incompatible resume accepted')
print('PASS: exact CPU optimizer/RNG/scheduler/scaler continuation, incompatible resume rejection, val/test isolation, metadata preflight')

from training_state import carry_best_weights
with tempfile.TemporaryDirectory() as tmp:
    source=Path(tmp)/'original';destination=Path(tmp)/'continued';source.mkdir()
    for name in ('best_iou.pth','best_psnr.pth'):(source/name).write_bytes(b'test-weight-bytes')
    carry_best_weights(source/'last.pt',destination)
    for name in ('best_iou.pth','best_psnr.pth'):assert (destination/name).read_bytes()==b'test-weight-bytes'
    (destination/'best_iou.pth').write_bytes(b'different')
    try:carry_best_weights(source/'last.pt',destination)
    except FileExistsError:pass
    else:raise AssertionError('Conflicting selected weights overwritten')
print('PASS: best-weight handoff and overwrite protection')
