import torch
from torch import nn
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from Model.CIFAR10.GoogLeNet.googlenet import googlenet

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="GoogLeNet(CIFAR-10)")
def create_model(is_pretrained=True, pretrained_file=None, no_normalize_layer=False, run_device=None):
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    if no_normalize_layer:
        googlenet_model = googlenet(pretrained=is_pretrained, pretrained_file=pretrained_file).to(run_device)
    else:
        googlenet_model = nn.Sequential(
            norm_layer,
            googlenet(pretrained=is_pretrained, pretrained_file=pretrained_file)
        ).to(device).eval()
    return googlenet_model