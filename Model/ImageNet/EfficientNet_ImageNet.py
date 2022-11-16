import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()

@sefi_component.model(name="EfficientNet(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    efficientnet_model = nn.Sequential(
        norm_layer,
        models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return efficientnet_model.eval()


@sefi_component.util(util_type="target_layers_getter", util_target="model", name="ResNet(ImageNet)")
def target_layers_getter(model):
    target_layers = [model[1].features]
    return target_layers
