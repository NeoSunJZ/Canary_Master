import base64
import json
from io import BytesIO

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tiia.v20190529 import tiia_client, models

from PIL import Image

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.model(name="TENCENT(API)", no_torch_model_check=True)
@sefi_component.config_params_handler(
    handler_target=ComponentType.MODEL, name="TENCENT(API)",
    handler_type=ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "secret_id": {"desc": "腾讯提供的secret_id", "type": "STR", "required": "true"},
        "secret_key": {"desc": "腾讯提供的secret_key", "type": "STR", "required": "true"},
        "endpoint": {"desc": "Endpoint", "type": "STR", "required": "true", "def":"tiia.tencentcloudapi.com"},
        "region_id": {"desc": "Region ID", "type": "STR", "required": "true", "def":"ap-beijing"},
    })
def create_model(secret_id, secret_key, endpoint="tiia.tencentcloudapi.com", region_id="ap-beijing", run_device=None):
    def image_recognition(base64_imgs):
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = endpoint

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = tiia_client.TiiaClient(cred, region_id, clientProfile)

        req = models.DetectLabelRequest()
        params = {
            "ImageBase64": base64_imgs,
            "Scenes": ["CAMERA"]
        }
        req.from_json_string(json.dumps(params))

        resp = client.DetectLabel(req)
        return resp.to_json_string()
    return image_recognition


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL, name="TENCENT(API)")
def img_pre_handler(ori_imgs, args):
    base64_imgs = []
    for ori_img in ori_imgs:
        buffer = BytesIO()
        pil_img = Image.fromarray(ori_img)
        pil_img.save(buffer, format="PNG")
        base64_imgs.append(base64.b64encode(buffer.getvalue()).rstrip().decode("utf-8"))
        del buffer
    return base64_imgs


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="TENCENT(API)")
def inference_detector(model, img):
    results = []
    for single_img in img:
        results.append(model(single_img))
    return results


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="TENCENT(API)")
def result_post_handler(result, args):
    predicts = []
    results = []
    for single_result in result:
        single_result = json.loads(single_result)
        score = {}
        for data in single_result["CameraLabels"]:
            score[data['Name']] = data['Confidence']
        results.append(score)
        predicts.append(single_result["CameraLabels"][0]['Name'])
    return predicts, results
