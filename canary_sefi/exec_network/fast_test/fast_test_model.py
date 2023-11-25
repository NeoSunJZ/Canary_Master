import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import Resize

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType
from canary_sefi.task_manager import task_manager

sefi_component = SEFIComponent()


@sefi_component.model(name="FAST-TEST-MODEL")
def create_model(model_file_name, run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = task_manager.base_temp_path + 'model/'
    model = torch.load(model_save_path + model_file_name).to(run_device).eval()
    return model


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL,
                     name=["FAST-TEST-MODEL"])
def img_pre_handler(ori_imgs, args):
    run_device = args.get("run_device", 'cuda' if torch.cuda.is_available() else 'cpu')
    result = None
    for ori_img in ori_imgs:
        ori_img = ori_img.copy().astype(np.float32)
        ori_img /= 255.0
        ori_img = ori_img.transpose(2, 0, 1)

        ori_img = Variable(torch.from_numpy(ori_img).to(run_device).float())

        # Resize
        img_size_h = args.get("img_size_h", 224)
        img_size_w = args.get("img_size_w", 224)
        resize = Resize([img_size_h, img_size_w])
        ori_img = resize(ori_img)
        ori_img = torch.unsqueeze(ori_img, dim=0)
        if result is None:
            result = ori_img
        else:
            result = torch.cat((result, ori_img), dim=0)
    return result


@sefi_component.util(util_type=SubComponentType.IMG_REVERSE_PROCESSOR, util_target=ComponentType.MODEL,
                     name=["FAST-TEST-MODEL"])

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


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL,
                     name=["FAST-TEST-MODEL"])
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return predicts, results


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL,
                     name=["FAST-TEST-MODEL"])
def inference_detector(model, img):
    model.eval()
    return model(img)
