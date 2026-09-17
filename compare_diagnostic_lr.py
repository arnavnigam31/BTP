"""Matched learning-rate continuations of a saved two-crop diagnostic (not a benchmark)."""
import argparse, gc, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from dataset import HySpecSegmentation, encode_segmap, get_labels
from models import model_generator
from utils import set_seed, init_mask, crop_mask, init_meas
from training_state import atomic_save, rng_state, restore_rng
from types import SimpleNamespace

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def summarize(hist, ce, mse):
    union=hist.sum(0)+hist.sum(1)-hist.diagonal()
    iou=np.divide(hist.diagonal(),union,out=np.zeros(len(hist)),where=union>0)
    present=hist.sum(1)>0;present[0]=False
    return {'ce':float(np.mean(ce)),'mse':float(np.mean(mse)),
            'foreground_miou_present':float(iou[present].mean()),
            'foreground_recall':float(hist.diagonal()[1:].sum()/hist[1:].sum()),
            'iou_by_class':iou.tolist(),'prediction_counts':hist.sum(0).tolist()}

def count_update(optimizer, scaler, loss):
    """Adam hook counts actual step calls, including AMP-skipped updates correctly."""
    count=[]
    handle=optimizer.register_step_post_hook(lambda *args:count.append(1))
    try:
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    finally:handle.remove()
    return len(count)

def preview(xs,ys,preds,selection,path):
    canvas=Image.new('RGB',(828,600),'white');draw=ImageDraw.Draw(canvas)
    for j,(x,y,pred) in enumerate(zip(xs,ys,preds)):
        rgb=x[:,:,[24,18,8]]
        rgb=(np.clip(rgb/max(float(np.quantile(rgb,.99)),1e-8),0,1)*255).astype(np.uint8)
        overlay=rgb.copy();colors=get_labels()[y].astype(np.uint8)
        overlay[y>0]=(.45*rgb[y>0]+.55*colors[y>0]).astype(np.uint8)
        for k,(im,title) in enumerate([(rgb,selection[j]['scene']), (overlay,'Ground truth overlay'),(get_labels()[pred].astype(np.uint8),'Prediction')]):
            canvas.paste(Image.fromarray(im),(k*276+10,j*300+30));draw.text((k*276+10,j*300+10),title,fill='black')
    canvas.save(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--from-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=10)
    a=p.parse_args()
    if a.steps<1 or a.eval_every<1:p.error('steps and eval-every must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    config=json.loads((a.from_run/'config.json').read_text())
    selection=json.loads((a.from_run/'selection.json').read_text())
    prior_metrics=json.loads((a.from_run/'metrics.json').read_text())[-1]
    checkpoint_path=a.from_run/'diagnostic_checkpoint.pt'
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    for key,value in {'transpose_image':True,'seed':3407,'lambda_rec':1,'lambda_seg':.0001,'fixed_mask':True}.items():
        if config.get(key)!=value or state['config'].get(key)!=value:raise ValueError('Unsupported source setting: '+key)
    if state['step']!=prior_metrics['step']:raise ValueError('Checkpoint and final metrics steps differ')
    if len(selection)!=2:raise ValueError('Exactly two saved training crops required')
    from run_public252 import validate
    spec_path=Path(__file__).with_name('public252-v1.json')
    validate(a.root,json.loads(spec_path.read_text()))
    loader=HySpecSegmentation(a.root,'train_data.csv',transpose_image=True)
    audited={r['scene']:r for r in json.loads((Path(__file__).parent/'research/dataset_audit_20260916/scenes.json').read_text())}
    label_hashes=json.loads((Path(__file__).parent/'research/dataset_audit_20260916/label_file_sha256.json').read_text())
    xs=[];ys=[]
    for sel in selection:
        name=sel['scene'];r=int(sel['row']);c=int(sel['col'])
        if name not in loader.names or r!=170 or not 0<=c<=256:raise ValueError('Invalid training crop selection')
        cube=np.load(a.root/'visible_28'/(name+'.npy'),allow_pickle=False)
        if hashlib.sha256(cube.tobytes()).hexdigest()!=audited[name]['sha256_cube28']:raise ValueError('Source cube changed')
        if digest(a.root/'labels'/(name+'.png'))!=label_hashes[name]:raise ValueError('Source label changed')
        image,label=loader.read_image_label(list(loader.names).index(name))
        xs.append(image[r:r+256,c:c+256].astype(np.float32));ys.append(encode_segmap(label)[r:r+256,c:c+256].copy())
    del cube,image,label
    set_seed(3407)
    mask_path=Path(__file__).parent/'mask/mask512x512.mat'
    phi,masks=init_mask(str(mask_path),'SSR',1,28)
    # Legacy code has exactly these two Python-random crop draws after set_seed(3407).
    phi,masks=crop_mask(phi,masks,SimpleNamespace(input_mask='SSR'),28)
    operator_hash=hashlib.sha256(phi.cpu().numpy().tobytes()).hexdigest()
    xgpu=[torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda() for x in xs]
    ygpu=[torch.from_numpy(y).long().unsqueeze(0).cuda() for y in ys]
    measurements=[init_meas(x,phi,'Y') for x in xgpu]
    a.out.mkdir(parents=True,exist_ok=False)
    provenance={'source_checkpoint_sha256':digest(checkpoint_path),'source_step':state['step'],
                'protocol_sha256':digest(spec_path),'mask_file_sha256':digest(mask_path),'cropped_phi_sha256':operator_hash,
                'additional_attempts_per_arm':a.steps,'evaluation_interval':a.eval_every,
                'rng_policy':'Same restored source RNG for both arms if available; otherwise same fresh seed 3407. Legacy checkpoint has no RNG: not bitwise continuation of original run.',
                'optimizer_policy':'Restore identical Adam moments and AMP scaler; change learning rate only.',
                'scope':'Training-crop diagnostic; not validation/test or novelty evidence',
                'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0)}
    (a.out/'comparison_config.json').write_text(json.dumps(provenance,indent=2))
    summaries={}
    for arm,lr in [('control_4e-4',.0004),('low_4e-5',.00004)]:
        out=a.out/arm;out.mkdir()
        set_seed(3407)
        model=model_generator('CRSDUN',28,23);model.load_state_dict(state['model'])
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        # Reload for each arm: optimizer.load_state_dict may retain tensor references.
        fresh=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        optimizer.load_state_dict(fresh['optimizer'])
        for group in optimizer.param_groups:group['lr']=lr
        scaler=torch.GradScaler('cuda');scaler.load_state_dict(fresh['scaler'])
        if 'rng' in fresh:restore_rng(fresh['rng'])
        else:set_seed(3407)
        del fresh
        history=[];updates=0;best=float('-inf');start=time.time()
        armconfig={**config,'learning_rate':lr,'initialization':'saved diagnostic checkpoint','additional_attempts':a.steps}
        (out/'config.json').write_text(json.dumps(armconfig,indent=2));(out/'selection.json').write_text(json.dumps(selection,indent=2))
        def evaluate(attempt):
            model.eval();hist=np.zeros((23,23),dtype=np.int64);ce=[];mse=[];preds=[]
            with torch.no_grad():
                for x,y,mea in zip(xgpu,ygpu,measurements):
                    with torch.autocast('cuda'):
                        xr,yr=model(mea,masks)
                        ce.append(float(torch.nn.functional.cross_entropy(yr[-1],y)))
                        mse.append(float(torch.nn.functional.mse_loss(xr[-1],x)))
                    if not torch.isfinite(xr[-1]).all() or not torch.isfinite(yr[-1]).all():raise ValueError('Nonfinite prediction')
                    pred=yr[-1].argmax(1).cpu().numpy()[0];preds.append(pred)
                    hist+=np.bincount((23*y.cpu().numpy()[0]+pred).ravel(),minlength=529).reshape(23,23)
            record={**summarize(hist,ce,mse),'step':int(state['step'])+attempt,'additional_attempts':attempt,
                    'effective_updates':updates,'skipped_updates':attempt-updates,'amp_scale':scaler.get_scale(),
                    'elapsed_seconds':time.time()-start}
            model.train();return record,preds
        def save(name,attempt):
            atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                         'step':int(state['step'])+attempt,'effective_updates_this_run':updates,'config':armconfig,
                         'rng':rng_state(),'selection':selection,'provenance':provenance},out/name)
        def record(attempt):
            nonlocal best
            row,preds=evaluate(attempt)
            if attempt==0:
                # Catch wrong masks/crops/checkpoints before spending GPU time. Allow small platform differences.
                for key,atol in [('ce',.005),('mse',.00002),('foreground_miou_present',.01)]:
                    if not np.isclose(row[key],prior_metrics[key],rtol=.02,atol=atol):
                        raise ValueError(f'Starting {key} differs from source: {row[key]} vs {prior_metrics[key]}; stop and inspect')
            history.append(row);(out/'metrics.json').write_text(json.dumps(history,indent=2))
            if row['foreground_miou_present']>best:
                best=row['foreground_miou_present'];save('best_checkpoint.pt',attempt);preview(xs,ys,preds,selection,out/'best_predictions.png')
            save('diagnostic_checkpoint.pt',attempt)
            preview(xs,ys,preds,selection,out/'predictions.png')
            print(json.dumps({'arm':arm,**row}),flush=True)
        record(0)
        for attempt in range(1,a.steps+1):
            j=(int(state['step'])+attempt-1)%2
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                xr,yr=model(measurements[j],masks)
                rec=sum(torch.nn.functional.mse_loss(xr[k],xgpu[j])*.7**i for i,k in enumerate(reversed(range(len(xr)))))
                seg=sum(torch.nn.functional.cross_entropy(yr[k],ygpu[j])*.7**i for i,k in enumerate(reversed(range(len(yr)))))
                loss=rec+.0001*seg
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            updates+=count_update(optimizer,scaler,loss)
            del xr,yr,rec,seg,loss
            if attempt%a.eval_every==0 or attempt==a.steps:record(attempt)
        tail=[r['foreground_miou_present'] for r in history[1:]][-10:]
        summaries[arm]={'initial':history[0],'final':history[-1],'best_logged_miou':best,
                        'last_10_logged_miou_mean':float(np.mean(tail)),'last_10_logged_miou_std':float(np.std(tail))}
        (a.out/'comparison_summary.json').write_text(json.dumps(summaries,indent=2))
        del record,evaluate,save,model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()
    for key in ['ce','mse','foreground_miou_present']:
        if not np.isclose(summaries['control_4e-4']['initial'][key],summaries['low_4e-5']['initial'][key],rtol=1e-5,atol=1e-6):
            raise ValueError('Comparison arms did not start equivalently')
    print('Matched comparison completed. Compare final AND recent mean/variability; do not select a method using a transient maximum alone.')

if __name__=='__main__':main()
