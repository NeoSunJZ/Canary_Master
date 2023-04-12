import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_lib.canary_model.cifar_10.MobileNetV2.mobilenetv2 import mobilenet_v2

sefi_component = SEFIComponent()


@sefi_component.model(name="MobileNetV2(CIFAR-10)")
def create_model(run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    mobilenet_model = nn.Sequential(
        norm_layer,
        mobilenet_v2(pretrained=True)
    ).to(run_device).eval()
    return mobilenet_model
