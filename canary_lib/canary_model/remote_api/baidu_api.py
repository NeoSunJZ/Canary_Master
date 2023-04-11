import base64
import json
from io import BytesIO

import requests
from PIL import Image

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.model(name="BAIDU(API)", no_torch_model_check=True)
@sefi_component.config_params_handler(
    handler_target=ComponentType.MODEL, name="BAIDU(API)",
    handler_type=ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "access_token": {"desc": "百度提供的access_token", "type": "STR", "required": "true"},
    })
def create_model(access_token, run_device=None):
    def image_recognition(base64_imgs):
        request_url = "https://aip.baidubce.com/api/v1/solution/direct/imagerecognition/combination"
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/json'}
        params = {
            "image": base64_imgs,
            "scenes": ["advanced_general"]
        }
        response = requests.post(request_url, data=json.dumps(params, indent=4, ensure_ascii=False), headers=headers)
        return response.text
    return image_recognition


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL, name="BAIDU(API)")
def img_pre_handler(ori_imgs, args):
    base64_imgs = []
    for ori_img in ori_imgs:
        buffer = BytesIO()
        pil_img = Image.fromarray(ori_img)
        pil_img.save(buffer, format="PNG")
        base64_imgs.append(base64.b64encode(buffer.getvalue()).rstrip().decode("utf-8"))
        del buffer
    return base64_imgs


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="BAIDU(API)")
def inference_detector(model, img):
    results = []
    for single_img in img:
        results.append(model(single_img))
    return results


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="BAIDU(API)")
def result_post_handler(result, args):
    predicts = []
    results = []
    for single_result in result:
        single_result = json.loads(single_result)
        score = {}
        for data in single_result["result"]["advanced_general"]["result"]:
            score[data['keyword']] = data['score']
        results.append(score)
        predicts.append(single_result["result"]["advanced_general"]["result"][0]['keyword'])
    return predicts, results
