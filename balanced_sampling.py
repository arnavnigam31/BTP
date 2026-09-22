"""Training-only 50:50 mixture of original sampling and foreground-class-targeted crops."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from dataset import encode_segmap, SegIdentityTransform, RandomHorizontalFlip

POLICY='balanced50_v1'

def eligible_columns(label, class_id):
    prefix=np.r_[0,np.cumsum((label[170:426]==class_id).sum(0))]
    columns=np.arange(10,246)
    counts=prefix[columns+256]-prefix[columns]
    maximum=int(counts.max())
    return columns[counts>=max(1,maximum*.5)].astype(np.int16) if maximum else np.array([],dtype=np.int16)

class BalancedTraining(Dataset):
    def __init__(self,base):
        self.base=base;self.names=base.names;self.num_classes=base.num_classes;self.class_names=base.class_names
        self.options={k:[] for k in range(1,self.num_classes)}
        self.labels=[]
        for i,name in enumerate(self.names):
            label=encode_segmap(np.asarray(Image.open(Path(base.root_dir)/'labels'/f'{name}.png'))).astype(np.uint8)
            if label.shape!=(512,512):raise ValueError('balanced50_v1 requires512x512 labels')
            self.labels.append(label)
            for k in np.unique(label):
                if not k:continue
                columns=eligible_columns(label,int(k))
                if len(columns):self.options[int(k)].append((i,columns))
        self.classes=sorted(k for k,v in self.options.items() if v)
        if len(self.classes)!=22:raise ValueError('Expected all22 foreground classes in training crop support')
        self.tensor=SegIdentityTransform();self.flip=RandomHorizontalFlip()

    def __len__(self):return len(self.base)

    def __getitem__(self,index):
        if np.random.random()<.5:
            sample=self.base[index]
            sample['sampling_target']=-1
            sample['sampling_scene_index']=index
            return sample
        target=self.classes[np.random.randint(len(self.classes))]
        choices=self.options[target];scene,cols=choices[np.random.randint(len(choices))]
        col=int(cols[np.random.randint(len(cols))])
        image,_=self.base.read_image_label(scene)
        label=self.labels[scene]
        sample=self.flip(self.tensor({'image':image[170:426,col:col+256].astype(float),
                                     'label':label[170:426,col:col+256].astype(int)}))
        sample['sampling_target']=target;sample['sampling_scene_index']=scene
        return sample
