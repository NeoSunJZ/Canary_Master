import torch
from torch import nn
from torchvision import models
from torchvision.models import Inception_V3_Weights
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.model(name="InceptionV3(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inception_model = nn.Sequential(
        norm_layer,
        models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return inception_model.eval()


@sefi_component.util(util_type="target_layers_getter", util_target="model", name="InceptionV3(ImageNet)")
def target_layers_getter(model):
    target_layers = [model[1].Mixed_7c]
    return target_layers, None