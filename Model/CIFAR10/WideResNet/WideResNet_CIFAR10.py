import torch
from torch import nn
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from Model.CIFAR10.WideResNet.wideresnet import wideresnet

sefi_component = SEFIComponent()


@sefi_component.model(name="WideResNet(CIFAR-10)")
def create_model(is_pretrained=True, pretrained_file=None, no_normalize_layer=False, run_device=None):
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if no_normalize_layer:
        wideresnet_model = wideresnet(pretrained=is_pretrained, pretrained_file=pretrained_file).to(run_device)
    else:
        wideresnet_model = nn.Sequential(
            norm_layer,
            wideresnet(pretrained=is_pretrained, pretrained_file=pretrained_file)
        ).to(run_device).eval()
    return wideresnet_model


@sefi_component.util(util_type="target_layers_getter", util_target="model", name="WideResNet(CIFAR-10)")
def target_layers_getter(model):
    target_layers = [model[1].features]
    return target_layers, None
