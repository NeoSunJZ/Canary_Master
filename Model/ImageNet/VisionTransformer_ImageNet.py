import torch
from torch import nn
from torchvision import models
from torchvision.models import ViT_B_32_Weights
from torchvision.transforms import Normalize, Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.model(name="ViT(ImageNet)")
def create_model(run_device):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    resize_layer = Resize([224, 224])
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    vit_model = nn.Sequential(
        resize_layer,
        norm_layer,
        models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)).to(run_device).eval()
    return vit_model.eval()


@sefi_component.util(util_type="target_layers_getter", util_target="model", name="ViT(ImageNet)")
def target_layers_getter(model):
    class ReshapeTransform:
        def __init__(self):
            input_size = [224, 224]
            patch_size = [32, 32]
            self.h = input_size[0] // patch_size[0]
            self.w = input_size[1] // patch_size[1]

        def __call__(self, x):
            # remove cls token and reshape
            # [batch_size, num_tokens, token_dim]
            result = x[:, 1:, :].reshape(x.size(0),
                                         self.h,
                                         self.w,
                                         x.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            # [batch_size, H, W, C] -> [batch, C, H, W]
            result = result.permute(0, 3, 1, 2)
            return result

    target_layers = [model[2].encoder.layers[-1].ln_1]
    return target_layers, ReshapeTransform()
