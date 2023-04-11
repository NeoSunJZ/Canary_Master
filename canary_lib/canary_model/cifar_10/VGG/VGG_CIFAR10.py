import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_lib.canary_model.cifar_10.VGG.vgg import vgg16_bn

sefi_component = SEFIComponent()


@sefi_component.model(name="VGG(CIFAR-10)")
def create_model(run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    vgg_model = nn.Sequential(
        norm_layer,
        vgg16_bn(pretrained=True)
    ).to(run_device).eval()
    return vgg_model
