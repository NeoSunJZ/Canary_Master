import os
import torch
from torch import nn
from torchvision.transforms import Normalize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_lib.canary_model.fashion_mnist.LeNetV5.lenet5 import LeNet5

sefi_component = SEFIComponent()


@sefi_component.model(name="LeNetV5(F-MNIST)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    lenet5 = LeNet5()
    script_dir = os.path.dirname(__file__)
    checkpoint = torch.load(
        script_dir + "/weight/best.pth.tar"
    )
    lenet5.load_state_dict(checkpoint['state_dict'])
    norm_layer = Normalize(mean=[0.406], std=[0.225])
    resnet19light_model = nn.Sequential(
        norm_layer,
        lenet5.to(run_device)
    )
    return resnet19light_model.to(run_device).eval()
