import torch
from torch import nn
from torchvision import models
from torchvision.models import MNASNet1_3_Weights
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.model(name="MNASNet(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    mnasnet_model = nn.Sequential(
        norm_layer,
        models.mnasnet1_3(weights=MNASNet1_3_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return mnasnet_model.eval()