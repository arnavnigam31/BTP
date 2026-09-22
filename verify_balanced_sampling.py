"""CPU alignment, deterministic replay, class support and unchanged baseline checks."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from dataset import get_labels,get_class_names,RandomCropHoriz,SegIdentityTransform,RandomHorizontalFlip
from balanced_sampling import BalancedTraining,eligible_columns

class Base:
 names=['fixture'];num_classes=23;class_names=get_class_names()
 def __len__(self):return 1
 def read_image_label(self,i):return self.label[:,:,None].astype(float),get_labels()[self.label].astype(np.uint8)
 def __getitem__(self,i):
  x,_=self.read_image_label(i)
  return RandomHorizontalFlip()(SegIdentityTransform()(RandomCropHoriz()({'image':x,'label':self.label.copy()})))

with tempfile.TemporaryDirectory() as tmp:
 b=Base();b.root_dir=tmp;b.label=np.zeros((512,512),dtype=np.uint8)
 for k in range(1,23):b.label[200:400,10+(k-1)*20:10+k*20]=k
 folder=Path(tmp)/'labels';folder.mkdir();Image.fromarray(get_labels()[b.label].astype(np.uint8)).save(folder/'fixture.png')
 np.random.seed(3);rng=np.random.get_state();ds=BalancedTraining(b)
 assert np.array_equal(rng[1],np.random.get_state()[1]) and rng[2]==np.random.get_state()[2]
 assert ds.classes==list(range(1,23))
 for seed in range(64):
  np.random.seed(seed);x=ds[0]
  assert np.array_equal(x['image'][0].numpy(),x['label'].numpy())
  np.random.seed(seed);repeat=ds[0]
  assert np.array_equal(x['label'].numpy(),repeat['label'].numpy())
  k=x['sampling_target']
  if k>0:
   maximum=max(int((b.label[170:426,c:c+256]==k).sum()) for c in range(10,246))
   assert int((x['label']==k).sum())>=maximum*.5
  else:
   np.random.seed(seed);np.random.random();expected=b[0]
   assert np.array_equal(x['label'].numpy(),expected['label'].numpy())
print('PASS:22-class support, aligned crops/flips, deterministic replay, no RNG consumed by index building, baseline crop branch unchanged.')
