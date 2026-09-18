"""Exact expected training-class exposure over all legal horizontal crop offsets."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from PIL import Image
from dataset import encode_segmap,get_class_names,RandomCropHoriz

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 names=pd.read_csv(a.root/'train_data.csv',index_col=0).names.astype(str).tolist()
 full=np.zeros(23,dtype=np.int64);band=full.copy();expected=np.zeros(23);scenes=np.zeros(23,dtype=int);prob=np.zeros(23)
 never=np.zeros(23,dtype=int);details=[]
 for name in names:
  y=encode_segmap(np.asarray(Image.open(a.root/'labels'/f'{name}.png')))
  assert y.shape==(512,512)
  full_hist=np.bincount(y.ravel(),minlength=23);band_hist=np.bincount(y[170:426].ravel(),minlength=23)
  counts=[]
  for k in range(23):
   columns=(y[170:426]==k).sum(axis=0)
   prefix=np.r_[0,np.cumsum(columns)]
   counts.append(prefix[np.arange(10,246)+256]-prefix[np.arange(10,246)])
  counts=np.asarray(counts)
  full+=full_hist;band+=band_hist;expected+=counts.mean(1);scenes+=full_hist>0;prob+=(counts>0).mean(1);never+=(full_hist>0)&(counts.max(1)==0)
  details.append({'scene':name,'full_pixels':full_hist.tolist(),'expected_crop_pixels':counts.mean(1).tolist(),'crop_presence_probability':(counts>0).mean(1).tolist()})
  # Confirm real sampler output agrees with its analytical offset distribution.
  if len(details)<=3:
   rng=np.random.get_state();np.random.seed(3407)
   np.random.randint(138,374);col=np.random.randint(138,374)-128
   np.random.seed(3407)
   got=RandomCropHoriz()({'image':np.zeros((512,512,1)), 'label':y})['label']
   np.random.set_state(rng)
   assert np.array_equal(got,y[170:426,col:col+256])
 rows=[]
 for k,name in enumerate(get_class_names()):
  rows.append({'id':k,'class':name,'training_scenes':int(scenes[k]),'full_pixels':int(full[k]),
   'fraction_pixels_in_vertical_band':float(band[k]/full[k]) if full[k] else None,
   'expected_pixels_per_epoch':float(expected[k]),'fraction_full_pixels_expected_per_epoch':float(expected[k]/full[k]) if full[k] else None,
   'expected_crops_with_class_per_epoch':float(prob[k]),'scenes_with_class_never_visible':int(never[k])})
 a.out.mkdir(parents=True,exist_ok=False)
 record={'scope':'Training labels only; no image cubes or validation/test pixels read',
   'sampler':'RandomCropHoriz256x256, row170:426, uniform column start10..245 inclusive; flip preserves counts',
   'meaning':'Exact expectation under sampler, not realized counts from past GPU runs',
   'train_csv_sha256':hashlib.sha256((a.root/'train_data.csv').read_bytes()).hexdigest(),'classes':rows,'scenes':details}
 (a.out/'exposure.json').write_text(json.dumps(record,indent=2)+'\n')
 for row in rows:print(json.dumps(row))

if __name__=='__main__':main()
