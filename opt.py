import argparse

parser = argparse.ArgumentParser(description="CRSDUN")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='fvgnet/', help='dataset directory')
parser.add_argument('--mask_path', type=str, default='mask/mask512x512.mat')

parser.add_argument('--transpose_image', action='store_true', help='Swap image spatial axes in memory to align public FVgNET candidate cubes with labels')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/CRSDUN/', help='saving_path')
parser.add_argument('--name', type=str, default='xxx_xxx', help='project name')

# Model specifications
parser.add_argument('--method', type=str, default='CRSDUN', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='Y',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='SSR',
                    help='the input mask of the network: SSR, Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')

parser.add_argument("--learning_rate", type=float, default=0.0004)

parser.add_argument('--resume', default=None, help='Full epoch-boundary last.pt checkpoint')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--eval_split', choices=['val', 'test'], default='val')
parser.add_argument('--stop_after_epoch', type=int, default=None, help='Stop after this absolute epoch, preserving max_epoch scheduler horizon')
opt = parser.parse_args()
if opt.stop_after_epoch is not None and not 1 <= opt.stop_after_epoch <= opt.max_epoch:
    parser.error('stop_after_epoch must be between 1 and max_epoch')
if opt.resume and opt.pretrained_model_path:
    parser.error('Choose resume or pretrained initialization, not both')

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
