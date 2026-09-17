"""CPU checks for the new diagnostic accounting and class metrics."""
import numpy as np
import torch
from compare_diagnostic_lr import count_update,summarize
torch.set_num_threads(2)
m=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(m.parameters(),lr=.001)
scaler=torch.GradScaler('cpu')
optimizer.zero_grad();loss=m(torch.ones(1,2)).square().mean()
assert count_update(optimizer,scaler,loss)==1
before={k:v.detach().clone() for k,v in m.state_dict().items()}
optimizer.zero_grad();loss=m(torch.ones(1,2)).sum()*float('inf')
assert count_update(optimizer,scaler,loss)==0
for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
hist=np.zeros((23,23),dtype=np.int64);hist[0,0]=10;hist[5,5]=6;hist[5,0]=2;hist[14,14]=2
r=summarize(hist,[.1,.3],[.01,.03])
assert abs(r['foreground_miou_present']-.875)<1e-12
assert abs(r['foreground_recall']-.8)<1e-12
assert abs(r['ce']-.2)<1e-12 and abs(r['mse']-.02)<1e-12
print('PASS: actual Adam updates counted, AMP overflow skips detected without parameter changes, present-class metrics correct')
