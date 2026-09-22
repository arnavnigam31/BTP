"""Single balanced-sampling arm against the archived epoch101-110 control."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,hashlib,json,statistics,subprocess,sys,shutil
import torch
from training_state import atomic_save,identity

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 repo=Path(__file__).resolve().parent;root=a.root.resolve();ref=a.reference.resolve();out=a.out.resolve()
 path=ref/'model/last.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!='637a9da66937c997d335df07b7f2491e228ad2ae452966a24175b13fc22936f5':raise ValueError('Use the verified epoch100 quarter-rate last.pt')
 state=torch.load(path,map_location='cpu',weights_only=True)
 expected=dict(method='CRSDUN',batch_size=1,max_epoch=500,learning_rate=.0001,workers=0,seed=3407,transpose_image=True,input_mask='SSR',input_setting='Y')
 for k,v in expected.items():
  if state['config'].get(k)!=v:raise ValueError(f'Unexpected reference setting {k}')
 assert state['epoch']==100 and state['scheduler']['eta_min']==2.5e-7
 assert state['config'].get('lambda_seg',1e-4)==1e-4 and state['config'].get('sampling_policy','baseline')=='baseline'
 assert identity(SimpleNamespace(data_root=root,mask_path=repo/'mask/mask512x512.mat'))==state['data_identity']
 subprocess.run([sys.executable,'run_public252.py','--root',str(root)],cwd=repo,check=True)
 control_path=repo/'research/sampling_control/comparison_summary.json'
 historical=json.loads(control_path.read_text())
 assert historical['experiment']['reference_checkpoint_sha256']==digest
 control=historical['arms']['control_seg1e-4']
 initial=json.loads((ref/'result/epoch_0100_training.json').read_text())
 state['config']['sampling_policy']='balanced50_v1';state['config']['lambda_seg']=1e-4
 state['best']={'iou':initial['val_foreground_miou_all22'],'psnr':initial['val_psnr_ref1']}
 out.mkdir(parents=True,exist_ok=False);seed=out/'initial_state';seed.mkdir()
 atomic_save(state,seed/'last.pt')
 for name in ['best_iou.pth','best_psnr.pth']:atomic_save(state['model'],seed/name)
 shutil.copytree(repo/'research/sampling_control',out/'historical_control')
 manifest={'reference_checkpoint_sha256':digest,'source_epoch':100,'end_epoch':110,
  'change':'50% original samples;50% uniformly chosen foreground class then uniform eligible training scene and legal class-rich horizontal crop',
  'policy':'balanced50_v1','control':'Archived control_seg1e-4 from paired_loss_20260918_141555; not concurrently rerun',
  'control_summary_sha256':hashlib.sha256(control_path.read_bytes()).hexdigest(),
  'constant':'quarter LR schedule, Adam/scaler starting state,lambda_seg1e-4,architecture,batch1,202 attempts per epoch',
  'caveat':'Sampling intentionally changes random draws and data sequence. CPU logging added to both code paths; no AMP/math/validation change.',
  'source_sha256':{f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in ['run_sampling_experiment.py','balanced_sampling.py','train.py','dataset.py','opt.py','training_state.py']}}
 (out/'experiment.json').write_text(json.dumps(manifest,indent=2))
 if a.prepare_only:print('Prepared verified epoch100 sampling branch; no training launched.');return
 subprocess.run([sys.executable,'-u','train.py','--data_root',str(root)+'/', '--transpose_image','--batch_size','1',
  '--workers','0','--seed','3407','--learning_rate','0.0001','--lambda_seg','0.0001','--sampling_policy','balanced50_v1',
  '--max_epoch','500','--stop_after_epoch','110','--resume',str(seed/'last.pt'),'--outf',str(out)+'/', '--name','balanced50'],cwd=repo,check=True)
 records=[json.loads((out/'balanced50/result'/f'epoch_{e:04d}_training.json').read_text()) for e in range(101,111)]
 values=[r['val_foreground_miou_all22'] for r in records]
 result={'final':records[-1],'epochs':records,'best_logged_continuation_miou':max(values),
  'mean_last5_miou':statistics.mean(values[-5:]),'std_last5_miou':statistics.pstdev(values[-5:]),
  'actual_updates':sum(r['optimizer_updates'] for r in records),'amp_skips':sum(r['amp_skipped_updates'] for r in records)}
 summary={'experiment':manifest,'arms':{'historical_control':control,'balanced50':result}}
 (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:{f:v for f,v in r.items() if f!='epochs'} for k,r in summary['arms'].items()},indent=2))

if __name__=='__main__':main()
