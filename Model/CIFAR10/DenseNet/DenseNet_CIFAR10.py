import torch
from torch import nn
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from Model.CIFAR10.DenseNet.densenet import densenet161

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="DenseNet(CIFAR-10)")
def create_model():
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    densenet_model = nn.Sequential(
        norm_layer,
        densenet161(pretrained=True)
    ).to(device).eval()
    return densenet_model.eval()