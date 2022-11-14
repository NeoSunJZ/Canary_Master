import torch
from torch import nn
from torchvision import models
from torchvision.models import ViT_B_32_Weights
from torchvision.transforms import Normalize, Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.model(name="ViT(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    resize_layer = Resize([224, 224])
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    vit_model = nn.Sequential(
        resize_layer,
        norm_layer,
        models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return vit_model.eval()