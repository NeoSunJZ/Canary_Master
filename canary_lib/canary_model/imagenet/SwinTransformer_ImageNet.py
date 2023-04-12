import math

import timm
import torch
from torch import nn
from torchvision import models
from torchvision.models import Swin_S_Weights
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.model(name="SwinTransformer(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    swin_s_model = nn.Sequential(
        norm_layer,
        models.swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return swin_s_model.eval()


@sefi_component.util(util_type=SubComponentType.MODEL_TARGET_LAYERS_GETTER, util_target=ComponentType.MODEL, name="SwinTransformer(ImageNet)")
def target_layers_getter(model):
    class ResizeTransform:
        def __init__(self):
            self.height = 7
            self.width = 7

        def __call__(self, x):
            x = x.flatten(start_dim=1, end_dim=2)
            result = x.reshape(x.size(0),
                               self.height,
                               self.width,
                               x.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            # [batch_size, H, W, C] -> [batch, C, H, W]
            result = result.permute(0, 3, 1, 2)

            return result
    target_layers = [model[1].norm]
    return target_layers, ResizeTransform()
