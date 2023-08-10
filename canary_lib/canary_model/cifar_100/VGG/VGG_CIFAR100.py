import torch
from torch import nn
from torchvision.transforms import Normalize

from .vgg import cifar100_vgg16_bn
from canary_sefi.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.model(name="VGG(CIFAR-100)")
def create_model(run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    vgg_model = nn.Sequential(
        norm_layer,
        cifar100_vgg16_bn(pretrained=True, device=run_device)
    ).to(run_device).eval()
    return vgg_model
