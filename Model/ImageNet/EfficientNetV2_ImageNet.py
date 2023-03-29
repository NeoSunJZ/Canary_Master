import torch
from torch import nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.transforms import Normalize, Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="EfficientNetV2(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    resize_layer = Resize([384, 384])
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    efficientnet_model = nn.Sequential(
        resize_layer,
        norm_layer,
        models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return efficientnet_model.eval()


@sefi_component.util(util_type=SubComponentType.MODEL_TARGET_LAYERS_GETTER, util_target=ComponentType.MODEL, name="EfficientNetV2(ImageNet)")
def target_layers_getter(model):
    target_layers = [model[2].features]
    return target_layers, None