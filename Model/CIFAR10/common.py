import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.util(util_type="img_preprocessor", util_target="model", name="DenseNet(CIFAR-10)")
def img_pre_handler(img, args):
    img = img.copy().astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img


@sefi_component.util(util_type="img_reverse_processor", util_target="model", name="DenseNet(CIFAR-10)")
def img_post_handler(adv, args):
    adv = np.squeeze(adv, axis=0)
    adv = adv.transpose(1, 2, 0)
    adv = adv * 255.0
    adv = np.clip(adv, 0, 255).astype(np.float32)
    return adv


@sefi_component.util(util_type="result_postprocessor", util_target="model", name="DenseNet(CIFAR-10)")
def result_post_handler(result, args):
    probs = F.softmax(result).detach().cpu().numpy()[0]
    pred = np.argmax(probs)
    return pred, probs


@sefi_component.util(util_type="inference_detector", util_target="model", name="DenseNet(CIFAR-10)")
def inference_detector(model, img):
    img_temp = Variable(torch.from_numpy(img).to(device).float())
    model.eval()
    return model(img_temp)
