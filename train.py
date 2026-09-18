import os
import json
from opt import opt
print(opt)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import math
import gc
import time
import torch
from torch import nn
from utils import *
from dataset import *
from models import *
from training_state import identity, record_run, save_state, load_state, atomic_save, carry_best_weights
scaler = torch.GradScaler(device="cuda")

import warnings
warnings.filterwarnings("ignore")

set_seed(opt.seed)

# dataset
train_loader, valid_loader = prep_loaders(
    root_dir=opt.data_root,
    transpose_image=opt.transpose_image,
    batch_size=opt.batch_size,
    workers=opt.workers
)

# mask
ch = 28
Phi_batch_train, input_mask_train = init_mask(opt.mask_path, opt.input_mask, opt.batch_size, ch)
Phi_batch_test, input_mask_test = init_mask(opt.mask_path, opt.input_mask, 1, ch)

# model
n_classes = train_loader.dataset.num_classes

model = model_generator(opt.method, ch, n_classes, opt.pretrained_model_path).cuda()

# optimizer
optimizer = torch.optim.Adam(params= model.parameters(),lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, 1e-6)

# loss
loss_fn_seg = nn.CrossEntropyLoss().cuda()
loss_fn_rec = nn.MSELoss().cuda()

# metrics
metrics_rec = Metrics_Rec()
metrics_seg = Metrics_Seg(train_loader.dataset.num_classes, train_loader.dataset.class_names)

# path
result_path = opt.outf + opt.name + '/result/'
model_path = opt.outf + opt.name + '/model/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

if os.path.exists(os.path.join(model_path, 'last.pt')) and not opt.resume:
    raise FileExistsError('Existing run: use --resume or choose a new name')
logger = gen_log(model_path)
data_id = identity(opt)
record_run(model_path, opt, data_id)

def main():
    best = {'iou': float('-inf'), 'psnr': float('-inf')}
    start_epoch = 0
    if opt.resume:
        start_epoch, best = load_state(opt.resume, model, optimizer, scheduler, scaler, opt, data_id)
        carry_best_weights(opt.resume, model_path)
        logger.info(f'Resuming after epoch {start_epoch}')
    lam_rec = 1
    lam_seg = opt.lambda_seg
    end_epoch = opt.stop_after_epoch if opt.stop_after_epoch is not None else opt.max_epoch
    if end_epoch <= start_epoch:
        raise ValueError('Stopping epoch must be greater than the resumed epoch')
    updates = [0]
    def count_update(*_):
        updates[0] += 1
    update_hook = optimizer.register_step_post_hook(count_update)
    for epoch in range(start_epoch, end_epoch):
        updates[0] = 0
        learning_rate = optimizer.param_groups[0]['lr']
        torch.cuda.reset_peak_memory_stats()
        model.train()

        # Progress reporting
        losses_rec = AverageMeter()
        losses_seg = AverageMeter()

        epoch_start = time.time()

        for batch_idx, sample in enumerate(train_loader):

            # Load a batch and send it to GPU
            x = sample['image'].float().cuda()
            y = sample['label'].long().cuda()

            Phi_syn, mask_in = crop_mask(Phi_batch_train, input_mask_train, opt, ch)
            mea = init_meas(x, Phi_syn, opt.input_setting)

            with torch.autocast(device_type="cuda", enabled=True):
                # Forward pass: compute predicted y by passing x to the model.
                x_pred_list, y_pred_list = model(mea, mask_in)

                # Compute and print loss.
                loss_rec = 0
                loss_seg = 0
                stage_list = list(range(len(x_pred_list)))
                stage_list.reverse()
                for stage_idx, stage in enumerate(stage_list):
                    loss_rec = loss_rec + loss_fn_rec(x_pred_list[stage], x) * math.pow(0.7, stage_idx)
                    loss_seg = loss_seg + loss_fn_seg(y_pred_list[stage], y) * math.pow(0.7, stage_idx)

                loss = lam_rec * loss_rec + lam_seg * loss_seg

            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, batch {batch_idx+1}')

            # Record loss
            losses_rec.update(loss_rec.data.item(), x.size(0))
            losses_seg.update(loss_seg.data.item(), y.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                remaining = avg_time * (len(train_loader) - batch_idx - 1)

                print(
                    f"Epoch {epoch+1}/{opt.max_epoch} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Rec {losses_rec.avg:.6f} | "
                    f"Seg {losses_seg.avg:.6f} | "
                    f"Elapsed {elapsed/60:.1f}m | "
                    f"ETA {remaining/60:.1f}m",
                    flush=True
                )

        logger.info(f"Rec loss: {losses_rec.avg}")
        logger.info(f"Seg loss: {losses_seg.avg}")
        logger.info(f"λ_rec: {lam_rec}")
        logger.info(f"λ_seg: {lam_seg}")

        scheduler.step()
        torch.cuda.empty_cache(); del x, y; gc.collect()

        # Validation after each epoch
        model.eval()
        metrics_rec.reset()
        metrics_seg.reset()
        for i, (sample) in enumerate(valid_loader):
            x, y = sample['image'].float().cuda(), sample['label'].numpy()
            mea = init_meas(x, Phi_batch_test, opt.input_setting)
            with torch.no_grad():
                x_pred_list, y_pred_list = model(mea, input_mask_test)
                x_pred = x_pred_list[-1]
                y_pred = y_pred_list[-1]
                y_pred = torch.argmax(y_pred, dim=1) # get the most likely prediction

            metrics_rec.add_batch(x.cpu(), x_pred.detach().cpu())
            metrics_seg.add_batch(y, y_pred.detach().cpu().numpy())
        metrics_rec_table = metrics_rec.get_table()
        metrics_seg_table = metrics_seg.get_table()
        logger.info(f"-------------------Epoch: {epoch+1}------------------------")
        logger.info(f'\nValidation stats:\n{metrics_rec_table}')
        logger.info(f'\nValidation stats:\n{metrics_seg_table}')
        # Save model
        val_iou = metrics_seg_table.at["total(-bg)", "IoU"]
        val_psnr = metrics_rec_table.at[0, "PSNR"]
        if val_iou > best['iou']:
            best['iou'] = float(val_iou)
            atomic_save(model.state_dict(), os.path.join(model_path, 'best_iou.pth'))
        if val_psnr > best['psnr']:
            best['psnr'] = float(val_psnr)
            atomic_save(model.state_dict(), os.path.join(model_path, 'best_psnr.pth'))
        metrics_rec_table.to_csv(os.path.join(result_path, f'epoch_{epoch+1:04d}_reconstruction.csv'), index=False)
        metrics_seg_table.to_csv(os.path.join(result_path, f'epoch_{epoch+1:04d}_segmentation.csv'))
        save_state(os.path.join(model_path, 'last.pt'), model, optimizer, scheduler, scaler,
                   epoch+1, best, opt, data_id)

        stats = {'epoch': epoch+1, 'scheduler_horizon': opt.max_epoch,
                 'lambda_rec': lam_rec, 'lambda_seg': lam_seg,
                 'learning_rate': learning_rate, 'attempted_updates': len(train_loader),
                 'optimizer_updates': updates[0], 'amp_skipped_updates': len(train_loader)-updates[0],
                 'train_reconstruction_loss': losses_rec.avg, 'train_segmentation_loss': losses_seg.avg,
                 'val_foreground_miou_all22': float(val_iou), 'val_psnr_ref1': float(val_psnr),
                 'elapsed_seconds': time.time()-epoch_start,
                 'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        with open(os.path.join(result_path, f'epoch_{epoch+1:04d}_training.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f'Epoch accounting: {stats}')
    update_hook.remove()
    print(f"Stopped after epoch {end_epoch}; scheduler horizon remains {opt.max_epoch}")

if __name__ == "__main__":
    "------------------start training-------------------------"
    main()
