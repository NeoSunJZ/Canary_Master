import torch
from torch import nn
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from Model.CIFAR10.DenseNet.densenet import densenet161

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="DenseNet(CIFAR-10)")
def create_model(is_pretrained=True, no_normalize_layer=False):
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if no_normalize_layer:
        densenet_model = densenet161(pretrained=is_pretrained).to(device)
    else:
        densenet_model = nn.Sequential(
            norm_layer,
            densenet161(pretrained=is_pretrained)
        ).to(device).eval()
    return densenet_model