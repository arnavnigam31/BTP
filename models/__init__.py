import torch
from .CRSDUN import CRSDUN

def model_generator(method, ch, n_classes, pretrained_model_path=None):
    if method == "CRSDUN":
        model = CRSDUN(stage=5, bands=ch, n_class=n_classes).cuda()

    else:
        raise ValueError(f'Unsupported model: {method}')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        load_weights(model, pretrained_model_path)
    return model

def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    weights = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict({k.removeprefix('module.'): v for k, v in weights.items()}, strict=True)
