"""Explicit, recorded validation/test evaluation for public252-v1."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dataset import evaluation_loader, decode_segmap
from utils import Metrics_Rec, Metrics_Seg, init_mask, init_meas, set_seed

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def evaluate_batches(model,loader,measurement,mask,device,out,limit=None,save_predictions=False,save_segmentation=False):
    """One scene per batch; measurement callback allows CPU integration fixtures."""
    rec=Metrics_Rec();seg=Metrics_Seg(loader.dataset.num_classes,loader.dataset.class_names)
    records=[];model.eval()
    with torch.inference_mode():
        for i,sample in enumerate(loader):
            if limit is not None and i>=limit:break
            target=sample['image'].to(device=device,dtype=torch.float32)
            truth=sample['label'].numpy()
            reconstructed,logits=model(measurement(target),mask)
            reconstructed=reconstructed[-1].float().cpu()
            predicted=logits[-1].argmax(dim=1).cpu()
            if reconstructed.shape!=target.shape or predicted.shape!=sample['label'].shape:
                raise ValueError('Model output shape differs from target')
            if not torch.isfinite(reconstructed).all() or not torch.isfinite(logits[-1]).all():
                raise ValueError('Nonfinite model output; evaluation aborted')
            scene=str(loader.dataset.names[i])
            one=Metrics_Rec();one.add_batch(target.cpu(),reconstructed)
            rec.add_batch(target.cpu(),reconstructed);seg.add_batch(truth,predicted.numpy())
            records.append({'scene':scene,**one.get_table().iloc[0].to_dict()})
            if save_segmentation:
                from PIL import ImageDraw
                from dataset import get_labels
                colors=get_labels().astype(np.uint8)
                ids=predicted.numpy()[0].astype(np.uint8)
                Image.fromarray(ids).save(out/(scene+'_class_ids.png'))
                hist=np.bincount((truth[0].astype(np.int64)*loader.dataset.num_classes+ids).ravel(),
                    minlength=loader.dataset.num_classes**2).reshape(loader.dataset.num_classes,loader.dataset.num_classes)
                np.save(out/(scene+'_confusion.npy'),hist)
                rgb=target[0,[24,18,8]].float().cpu().permute(1,2,0).numpy()
                rgb=(np.clip(rgb/max(float(np.quantile(rgb,.99)),1e-8),0,1)*255).astype(np.uint8)
                panels=[rgb,colors[truth[0]],colors[ids]]
                canvas=Image.new('RGB',(3*rgb.shape[1],rgb.shape[0]+30),'white')
                draw=ImageDraw.Draw(canvas)
                for j,(panel,label) in enumerate(zip(panels,['Pseudo-RGB','Ground truth','Prediction'])):
                    canvas.paste(Image.fromarray(panel),(j*rgb.shape[1],30))
                    draw.text((j*rgb.shape[1]+5,8),label,fill='black')
                canvas.save(out/(scene+'_preview.png'))
            if save_predictions:
                np.save(out/(scene+'_hsi.npy'),reconstructed[0].permute(1,2,0).numpy())
                Image.fromarray(decode_segmap(predicted).astype(np.uint8)).save(out/(scene+'.png'))
            print(f'Evaluated {len(records)}/{min(len(loader),limit or len(loader))}',flush=True)
    if not records:raise ValueError('No evaluation scenes')
    import pandas as pd
    rec.get_table().to_csv(out/'reconstruction.csv',index=False)
    seg.get_table().to_csv(out/'segmentation.csv')
    pd.DataFrame(records).to_csv(out/'per_scene_reconstruction.csv',index=False)
    np.save(out/'confusion_matrix.npy',seg.confusion_matrix)
    return [r['scene'] for r in records]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root',type=Path,required=True)
    parser.add_argument('--pretrained_model_path',type=Path,required=True)
    parser.add_argument('--eval_split',choices=['val','test'],default='val')
    parser.add_argument('--transpose_image',action='store_true')
    parser.add_argument('--outf',type=Path,default=Path('exp/evaluation'))
    parser.add_argument('--name',default='public252_v1')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--mask_path',type=Path,default=Path(__file__).resolve().parent/'mask/mask512x512.mat')
    parser.add_argument('--limit',type=int,help='Validation-only infrastructure smoke check; never a benchmark')
    parser.add_argument('--save_predictions',action='store_true')
    parser.add_argument('--save_segmentation',action='store_true',help='Save class IDs, per-scene confusion matrices and previews without HSI cubes')
    parser.add_argument('--batch_size',type=int,default=1,help='Evaluation requires one scene per batch')
    args=parser.parse_args()
    if args.batch_size!=1 or args.workers<0:parser.error('Use batch_size=1 and nonnegative workers')
    if Path(args.name).name!=args.name or args.name in ('.','..'):parser.error('name must be a single folder name')
    if args.limit is not None and (args.limit<1 or args.eval_split!='val'):
        parser.error('--limit is positive and permitted only for validation smoke checks')
    if not args.transpose_image:parser.error('public252-v1 requires --transpose_image')
    from run_public252 import validate
    spec_path=Path(__file__).with_name('public252-v1.json')
    validate(args.data_root,json.loads(spec_path.read_text()))
    if not args.pretrained_model_path.is_file():parser.error('Checkpoint file does not exist')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for the CRSDUN measurement operator')
    out=args.outf/args.name/('smoke_val' if args.limit else 'evaluation_'+args.eval_split)
    out.mkdir(parents=True,exist_ok=False)
    set_seed(3407)
    from models import model_generator
    loader=evaluation_loader(args.data_root,args.eval_split,args.workers,True)
    model=model_generator('CRSDUN',28,loader.dataset.num_classes,str(args.pretrained_model_path))
    phi,mask=init_mask(str(args.mask_path),'SSR',1,28)
    record={'status':'running','split':args.eval_split,'scope':'validation smoke' if args.limit else 'full split',
            'checkpoint_sha256':sha256(args.pretrained_model_path),'mask_sha256':sha256(args.mask_path),
            'protocol_sha256':sha256(spec_path),'reference_amplitude':1.0,'seed':3407,
            'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
            'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (out/'evaluation.json').write_text(json.dumps(record,indent=2))
    scenes=evaluate_batches(model,loader,lambda x:init_meas(x,phi,'Y'),mask,'cuda',out,args.limit,args.save_predictions,args.save_segmentation)
    record.update(status='completed',scene_count=len(scenes),scenes=scenes)
    (out/'evaluation.json').write_text(json.dumps(record,indent=2))
    print(f'Completed {args.eval_split}: {len(scenes)} scenes. Results: {out}')

if __name__=='__main__':main()
