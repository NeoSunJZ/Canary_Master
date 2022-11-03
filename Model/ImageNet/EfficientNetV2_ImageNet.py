import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_L_Weights
from torchvision.transforms import Normalize, Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="EfficientNetV2(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is None else ('cuda' if torch.cuda.is_available() else 'cpu')
    resize_layer = Resize([480, 480])
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    efficientnet_model = nn.Sequential(
        resize_layer,
        norm_layer,
        models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return efficientnet_model.eval()