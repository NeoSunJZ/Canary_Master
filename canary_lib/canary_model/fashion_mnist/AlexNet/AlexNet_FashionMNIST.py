import os
import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_lib.canary_model.fashion_mnist.AlexNet.alexnet import AlexNetLight

sefi_component = SEFIComponent()


@sefi_component.model(name="AlexNet(F-MNIST)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    alexnet19light = AlexNetLight()
    script_dir = os.path.dirname(__file__)
    checkpoint = torch.load(
        script_dir + "/weight/best.pth.tar"
    )
    alexnet19light.load_state_dict(checkpoint['state_dict'])
    norm_layer = Normalize(mean=[0.406], std=[0.225])
    resnet19light_model = nn.Sequential(
        norm_layer,
        alexnet19light.to(run_device)
    )
    return resnet19light_model.to(run_device).eval()
