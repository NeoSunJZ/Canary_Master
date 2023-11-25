import base64
import json
from io import BytesIO

import numpy as np
import requests
import torch
from torch import nn
from torchvision import transforms
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType, ComponentConfigHandlerType
from torch.autograd import Variable
from torchvision.transforms import Resize

sefi_component = SEFIComponent()


@sefi_component.model(name="RemoteModel")
@sefi_component.config_params_handler(handler_target=ComponentType.MODEL, name="RemoteModel",
                                      handler_type=ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                          "url": {"desc": "远程模型API地址", "type": "STR", "required": "true"},
                                     })
def create_model(url, run_device=None):
    run_device = run_device if run_device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        RemoteModel(run_device, url)
    ).to(run_device).eval()
    return model


class RemoteModel(nn.Module):
    def __init__(self, run_device, url) -> None:
        super().__init__()
        self.run_device = run_device
        self.url = url

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = transforms.ToPILImage()
        outputs_array = []
        for img in x:
            pil_image = transform(img)
            output_buffer = BytesIO()
            pil_image.save(output_buffer, format='png')
            image_data = output_buffer.getvalue()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            params = {
                "base64": "data:image/png;base64," + base64_string
            }
            response = requests.post(self.url, json=params)
            outputs = json.loads(response.text).get("outputs")
            outputs_array.append(outputs)
        outputs = torch.Tensor(outputs_array).to(self.run_device)
        return outputs


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL,
                     name=["RemoteModel"])
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
                     name=["RemoteModel"])

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
                     name=["RemoteModel"])
def result_post_handler(logits, args):
    results = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
    predicts = []
    for result in results:
        predicts.append(np.argmax(result))
    return predicts, results


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL,
                     name=["RemoteModel"])
def inference_detector(model, img):
    model.eval()
    return model(img)
