import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import Resize

from CANARY_SEFI.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@sefi_component.util(util_type="img_preprocessor", util_target="model", name="DenseNet(CIFAR-10)")
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)
        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())

        # Resize
        img_size_h = args.get("img_size_h", 32)
        img_size_w = args.get("img_size_w", 32)
        resize = Resize([img_size_h, img_size_w])
        ori_img = resize(ori_img)
        ori_img = torch.unsqueeze(ori_img, dim=0)
        if result is None:
            result = ori_img
        else:
            result = torch.cat((result, ori_img), dim=0)
    return result

    # img = img.copy().astype(np.float32)
    # img /= 255.0
    # img = img.transpose(2, 0, 1)
    # img = torch.from_numpy(np.expand_dims(img, axis=0))
    # return img


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
    return pred, prob


@sefi_component.inference_detector(model_name="DenseNet(CIFAR-10)", support_type="numpy_array", return_type="label_string")
def inference_detector(model, img):
    img_temp = Variable(torch.from_numpy(img).to(device).float())
    model.eval()
    return model(img_temp)
