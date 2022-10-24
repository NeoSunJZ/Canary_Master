import cv2
import torch
from numpy.linalg import norm
from torch import nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
from torchvision.models import AlexNet_Weights
import torch.nn.functional as F
from torchvision.transforms import Normalize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="Alexnet(ImageNet)")
def create_model():
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
    alexnet_model = nn.Sequential(
        norm_layer,
        models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    ).to(device).eval()
    return alexnet_model.eval()


@sefi_component.util(util_type="img_preprocessor", util_target="model", name="Alexnet(ImageNet)")
def img_pre_handler(img, args):
    img = img.copy().astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="Alexnet(ImageNet)")
def img_post_handler(adv, args):
    adv = np.squeeze(adv, axis=0)
    adv = adv.transpose(1, 2, 0)
    adv = adv * 255.0
    adv = np.clip(adv, 0, 255).astype(np.float32)
    return adv


@sefi_component.util(util_type="result_postprocessor", util_target="model", name="Alexnet(ImageNet)")
def result_post_handler(result, args):
    probs = F.softmax(result).detach().cpu().numpy()[0]
    pred = np.argmax(probs)
    return pred, probs


@sefi_component.inference_detector(model_name="Alexnet(ImageNet)", support_type="numpy_array",
                                   return_type="label_string")
def inference_detector(model, img):
    img_temp = Variable(torch.from_numpy(img).to(device).float())
    model.eval()
    return model(img_temp)
