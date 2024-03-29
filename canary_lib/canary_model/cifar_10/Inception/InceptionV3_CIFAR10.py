import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_lib.canary_model.cifar_10.Inception.inception import inception_v3
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.model(name="InceptionV3(CIFAR-10)")
def create_model(is_pretrained=True, pretrained_file=None, no_normalize_layer=False, run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if no_normalize_layer:
        inception_model = inception_v3(pretrained=is_pretrained, pretrained_file=pretrained_file).to(run_device)
    else:
        inception_model = nn.Sequential(
            norm_layer,
            inception_v3(pretrained=is_pretrained, pretrained_file=pretrained_file)
        ).to(run_device).eval()
    return inception_model


@sefi_component.util(util_type=SubComponentType.MODEL_TARGET_LAYERS_GETTER, util_target=ComponentType.MODEL, name="InceptionV3(CIFAR-10)")
def target_layers_getter(model):
    target_layers = [model[1].Mixed_7c]
    return target_layers, None