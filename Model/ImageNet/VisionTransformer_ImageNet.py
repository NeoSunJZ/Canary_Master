import torch
from torch import nn
from torchvision import models
from torchvision.models import ViT_H_14_Weights
from torchvision.transforms import Normalize, Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="ViT(ImageNet)")
def create_model():
    resize_layer = Resize([518,518])
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    densenet_model = nn.Sequential(
        resize_layer,
        norm_layer,
        models.vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)).to(device).eval()
    return densenet_model.eval()