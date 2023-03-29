import torch
from torch import nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.model(name="GoogLeNet(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    googlenet_model = nn.Sequential(
        norm_layer,
        models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return googlenet_model.eval()


@sefi_component.util(util_type=SubComponentType.MODEL_TARGET_LAYERS_GETTER, util_target=ComponentType.MODEL, name="GoogLeNet(ImageNet)")
def target_layers_getter(model):
    target_layers = [model[1].inception5b]
    return target_layers, None