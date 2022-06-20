import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
from torchvision.models import ViT_L_16_Weights

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.model(name="VisionTransformer")
def create_model():
    vit_l_16 = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device).eval()
    return vit_l_16


@sefi_component.util(util_type="img_preprocessor", util_target="model", name="VisionTransformer")
def img_pre_handler(img, args):
    img = img.copy().astype(np.float32)
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    img /= 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0)
    return img


@sefi_component.util(util_type="result_postprocessor", util_target="model", name="VisionTransformer")
def result_post_handler(result, args):
    orig_label = np.argmax(result)
    return "orig_label={}".format(orig_label)


@sefi_component.inference_detector(model_name="VisionTransformer", support_type="numpy_array",
                                   return_type="label_string")
def inference_detector(model, img):
    # 判断设备

    img_temp = Variable(torch.from_numpy(img).to(device).float())
    result = model(img_temp).data.to(device).numpy()
    return result
