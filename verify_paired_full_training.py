"""Check preserved Adam/RNG state and quarter-scaled LR through resumed steps on CPU."""
import copy
import torch
from paired_full_training import branch_state

def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a,b)
    if isinstance(a, dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

torch.manual_seed(3407)
m=torch.nn.Linear(2,1)
opt=torch.optim.Adam(m.parameters(),lr=.0004)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,500,1e-6)
for _ in range(50):
    opt.zero_grad();m(torch.ones(1,2)).square().sum().backward();opt.step();sched.step()
source={'config':{'learning_rate':.0004},'model':m.state_dict(),'optimizer':opt.state_dict(),
        'scheduler':sched.state_dict(),'rng':{'torch':torch.get_rng_state()},'scaler':{'scale':32768},
        'best':{'iou':.44,'psnr':33.54},'epoch':50}
before=copy.deepcopy(source)
control=branch_state(source,1.,{'iou':.22,'psnr':30.4})
quarter=branch_state(source,.25,{'iou':.22,'psnr':30.4})
assert equal(source,before), 'Source mutated'
for field in ['model','rng','scaler','epoch']:
    assert equal(control[field],quarter[field]) and equal(control[field],source[field])
assert equal(control['optimizer']['state'],quarter['optimizer']['state'])
assert equal(control['optimizer'],source['optimizer'])
assert equal(control['scheduler'],source['scheduler'])
optimizers=[];schedulers=[]
for state in [control,quarter]:
    model=torch.nn.Linear(2,1);model.load_state_dict(state['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=state['config']['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,500,1e-6)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    optimizers.append(optimizer);schedulers.append(scheduler)
for _ in range(10):
    assert abs(optimizers[1].param_groups[0]['lr']-.25*optimizers[0].param_groups[0]['lr'])<1e-15
    for optimizer,scheduler in zip(optimizers,schedulers): optimizer.step();scheduler.step()
assert schedulers[0].last_epoch==schedulers[1].last_epoch==60
print('PASS: original state unchanged; model/Adam moments/scaler/RNG retained; quarter LR preserved through epochs51-60; original scheduler horizon retained.')
