import torch
from torch import nn
from torchvision import models
from torchvision.models import ShuffleNet_V2_X2_0_Weights
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.model(name="ShuffleNetV2(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    shufflenet_model = nn.Sequential(
        norm_layer,
        models.shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return shufflenet_model.eval()


@sefi_component.util(util_type=SubComponentType.MODEL_TARGET_LAYERS_GETTER, util_target=ComponentType.MODEL, name="ShuffleNetV2(ImageNet)")
def target_layers_getter(model):
    target_layers = [model[1].conv5]
    return target_layers, None