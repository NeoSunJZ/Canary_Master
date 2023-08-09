import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip, Normalize, Compose

from canary_sefi.core.component.component_decorator import SEFIComponent

sefi_component = SEFIComponent()


@sefi_component.util(util_type="img_preprocessor", util_target="model",
                     name=["DenseNet(CIFAR-10)", "GoogLeNet(CIFAR-10)", "InceptionV3(CIFAR-10)",
                           "MobileNetV2(CIFAR-10)",
                           "ResNet(CIFAR-10)", "VGG(CIFAR-10)"])
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)

        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())

        # Resize
        Resize_args = args.get("Resize", {"size": [32, 32]})
        Resize_handler = Resize(**Resize_args)
        ori_img = Resize_handler(ori_img)

        transforms_args = args.get("transforms", None)
        if transforms_args is not None:
            compose_handler = Compose([eval(key)(**transforms_args[key]) for key in transforms_args])
            ori_img = compose_handler(ori_img)

        ori_img = torch.unsqueeze(ori_img, dim=0)
        if result is None:
            result = ori_img
        else:
            result = torch.cat((result, ori_img), dim=0)
    return result


@sefi_component.util(util_type="img_reverse_processor", util_target="model",
                     name=["DenseNet(CIFAR-10)", "GoogLeNet(CIFAR-10)", "InceptionV3(CIFAR-10)",
                           "MobileNetV2(CIFAR-10)",
                           "ResNet(CIFAR-10)", "VGG(CIFAR-10)"])
def img_post_handler(adv_imgs, args):
    if type(adv_imgs) == torch.Tensor:
        adv_imgs = adv_imgs.data.cpu().numpy()

    result = []
    for adv_img in adv_imgs:
        adv_img = adv_img.transpose(1, 2, 0)
        adv_img = adv_img * 255.0
        adv_img = np.clip(adv_img, 0, 255).astype(np.float32)
        result.append(adv_img)
    return result


@sefi_component.util(util_type="result_postprocessor", util_target="model",
                     name=["DenseNet(CIFAR-10)", "GoogLeNet(CIFAR-10)", "InceptionV3(CIFAR-10)",
                           "MobileNetV2(CIFAR-10)",
                           "ResNet(CIFAR-10)", "VGG(CIFAR-10)"])
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return predicts, results


@sefi_component.util(util_type="inference_detector", util_target="model",
                     name=["DenseNet(CIFAR-10)", "GoogLeNet(CIFAR-10)", "InceptionV3(CIFAR-10)",
                           "MobileNetV2(CIFAR-10)",
                           "ResNet(CIFAR-10)", "VGG(CIFAR-10)"])
def inference_detector(model, img):
    model.eval()
    return model(img)
