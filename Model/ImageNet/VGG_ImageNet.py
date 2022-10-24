import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_BN_Weights
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="VGG(ImageNet)")
def create_model():
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    densenet_model = nn.Sequential(
        norm_layer,
        models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)).to(device).eval()
    return densenet_model.eval()