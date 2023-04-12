import numpy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL,
                     name=["AlexNet(F-MNIST)", "ResNet(F-MNIST)", "LeNetV5(F-MNIST)"])
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)
        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())
        ori_img = torch.unsqueeze(ori_img, dim=0)
        ori_img = torch.unsqueeze(ori_img, dim=0)
        if result is None:
            result = ori_img
        else:
            result = torch.cat((result, ori_img), dim=0)
    return result


@sefi_component.util(util_type=SubComponentType.IMG_REVERSE_PROCESSOR, util_target=ComponentType.MODEL,
                     name=["AlexNet(F-MNIST)", "ResNet(F-MNIST)", "LeNetV5(F-MNIST)"])
def img_post_handler(adv_imgs, args):
    if type(adv_imgs) == torch.Tensor:
        adv_imgs = adv_imgs.data.cpu().numpy()

    result = []
    for adv_img in adv_imgs:
        adv_img = np.clip(adv_img, 0, 255).astype(np.float32)
        result.append(adv_img)
    return result


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL,
                     name=["AlexNet(F-MNIST)", "ResNet(F-MNIST)", "LeNetV5(F-MNIST)"])
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return (predicts, results)


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL,
                     name=["AlexNet(F-MNIST)", "ResNet(F-MNIST)", "LeNetV5(F-MNIST)"])
def inference_detector(model, img):
    model.eval()
    return model(img)
