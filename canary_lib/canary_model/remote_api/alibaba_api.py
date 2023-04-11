import os

from PIL import Image
from alibabacloud_imagerecog20190930.client import Client
from alibabacloud_imagerecog20190930.models import TaggingImageAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType, ComponentConfigHandlerType
from canary_sefi.task_manager import task_manager

sefi_component = SEFIComponent()


@sefi_component.model(name="ALIBABA(API)", no_torch_model_check=True)
@sefi_component.config_params_handler(
    handler_target=ComponentType.MODEL, name="ALIBABA(API)",
    handler_type=ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "access_key_id": {"desc": "您的 AccessKey ID(生成请参考https://help.aliyun.com/document_detail/175144.html)", "type": "STR", "required": "true"},
        "access_key_secret": {"desc": "您的 AccessKey Secret(生成请参考https://help.aliyun.com/document_detail/175144.html)", "type": "STR", "required": "true"},
        "endpoint": {"desc": "Endpoint", "type": "STR", "required": "true", "def": "imagerecog.cn-shanghai.aliyuncs.com"},
        "region_id": {"desc": "Region ID", "type": "STR", "required": "true", "def": "cn-shanghai"}
    })
def create_model(access_key_id, access_key_secret, endpoint, region_id, run_device=None):
    def image_recognition(img):
        # 使用ByteIO的buffer会导致奇怪的错误，折中修复方案
        temp_file_name = task_manager.base_temp_path + "temp"
        Image.fromarray(img).save(temp_file_name, format='PNG')
        img = open(temp_file_name, 'rb')

        tagging_image_request = TaggingImageAdvanceRequest()
        tagging_image_request.image_urlobject = img
        runtime = RuntimeOptions(read_timeout=50000)
        client = Client(Config(access_key_id=access_key_id, access_key_secret=access_key_secret,
                               endpoint=endpoint, region_id=region_id))
        response = client.tagging_image_advance(tagging_image_request, runtime)
        os.remove(temp_file_name)
        return response.body
    return image_recognition


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="ALIBABA(API)")
def inference_detector(model, img):
    results = []
    for single_img in img:
        results.append(model(single_img))
    return results


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="ALIBABA(API)")
def result_post_handler(result, args):
    predicts = []
    results = []
    for single_result in result:
        score = {}
        for data in single_result.data.tags:
            score[data.value] = data.confidence
        results.append(score)
        predicts.append(single_result.data.tags[0].value)
    return predicts, results
