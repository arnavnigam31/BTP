"""CPU integration checks for the production evaluator and checkpoint formats."""
import importlib.util,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from models import load_weights
from utils import Metrics_Rec
spec=importlib.util.spec_from_file_location('production_evaluator',Path(__file__).with_name('test.py'))
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
torch.set_num_threads(2)
class Scenes(Dataset):
    names=['fixture_a','fixture_b'];num_classes=3;class_names=['bg','one','two']
    def __len__(self):return 2
    def __getitem__(self,i):
        return {'image':torch.linspace(0,2,28*16*16).reshape(28,16,16),
                'label':torch.full((16,16),i+1,dtype=torch.long)}
class Model(torch.nn.Module):
    def forward(self,x,mask):
        logits=torch.zeros(len(x),3,16,16);logits[:,1]=1
        return [x+.1],[logits]
with tempfile.TemporaryDirectory() as temporary:
    out=Path(temporary)
    names=evaluator.evaluate_batches(Model(),DataLoader(Scenes(),batch_size=1),lambda x:x,None,'cpu',out)
    assert names==Scenes.names
    table=pd.read_csv(out/'reconstruction.csv')
    assert abs(table.MSE.iloc[0]-.01)<1e-6 and abs(table.PSNR.iloc[0]-20)<1e-4
    matrix=np.load(out/'confusion_matrix.npy')
    assert matrix[1,1]==256 and matrix[2,1]==256 and matrix.sum()==512
    assert len(pd.read_csv(out/'per_scene_reconstruction.csv'))==2
    assert not list(out.glob('*_hsi.npy'))
    export=out/'segmentation_only';export.mkdir()
    evaluator.evaluate_batches(Model(),DataLoader(Scenes(),batch_size=1),lambda x:x,None,'cpu',export,save_segmentation=True)
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(export/'fixture_a_class_ids.png')),np.ones((16,16),dtype=np.uint8))
    assert np.array_equal(np.load(export/'fixture_a_confusion.npy')+np.load(export/'fixture_b_confusion.npy'),matrix)
    assert Image.open(export/'fixture_a_preview.png').size==(48,46)
    assert not list(export.glob('*_hsi.npy'))
    metric=Metrics_Rec();target=torch.ones(1,28,16,16)
    metric.add_batch(target,target+.1);metric.reset();metric.add_batch(target,target+.2)
    assert abs(metric.get_table().MSE.iloc[0]-.04)<1e-6
    class Broken(Model):
        def forward(self,x,mask):
            r,s=super().forward(x,mask);r[0][0,0,0,0]=float('nan');return r,s
    try:evaluator.evaluate_batches(Broken(),DataLoader(Scenes()),lambda x:x,None,'cpu',out)
    except ValueError as error:assert 'Nonfinite' in str(error)
    else:raise AssertionError('Invalid predictions accepted')
    original=torch.nn.Linear(3,2)
    for name,state in [('plain',original.state_dict()),('ddp',{'module.'+k:v for k,v in original.state_dict().items()}),('full',{'model':original.state_dict(),'epoch':3})]:
        path=out/(name+'.pt');torch.save(state,path)
        recovered=torch.nn.Linear(3,2);load_weights(recovered,path)
        for key,value in original.state_dict().items():torch.testing.assert_close(value,recovered.state_dict()[key])
print('PASS: production CPU evaluation, MSE/PSNR, confusion matrix, scene IDs, no default cubes, nonfinite rejection and three checkpoint formats')
