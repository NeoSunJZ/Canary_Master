import base64
from io import BytesIO

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkimage.v2 import ImageClient, RunImageTaggingRequest, ImageTaggingReq
from huaweicloudsdkimage.v2.region.image_region import ImageRegion

from PIL import Image

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import SubComponentType, ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.model(name="HUAWEI(API)", no_torch_model_check=True)
@sefi_component.config_params_handler(
    handler_target=ComponentType.MODEL, name="HUAWEI(API)",
    handler_type=ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
    use_default_handler=True,
    params={
        "ak": {"desc": "华为提供的ak", "type": "STR", "required": "true"},
        "sk": {"desc": "华为提供的sk", "type": "STR", "required": "true"},
    })
def create_model(ak, sk, run_device=None):
    def image_recognition(base64_imgs):
        credentials = BasicCredentials(ak, sk)
        client = ImageClient.new_builder() \
            .with_credentials(credentials) \
            .with_region(ImageRegion.value_of("cn-north-1")) \
            .build()
        request = RunImageTaggingRequest()
        request.body = ImageTaggingReq(
            limit=50, threshold=95, language="zh", image=base64_imgs)
        response = client.run_image_tagging(request)
        return response
    return image_recognition


@sefi_component.util(util_type=SubComponentType.IMG_PREPROCESSOR, util_target=ComponentType.MODEL, name="HUAWEI(API)")
def img_pre_handler(ori_imgs, args):
    base64_imgs = []
    for ori_img in ori_imgs:
        buffer = BytesIO()
        pil_img = Image.fromarray(ori_img)
        pil_img.save(buffer, format="PNG")
        base64_imgs.append(base64.b64encode(buffer.getvalue()).rstrip().decode("utf-8"))
        del buffer
    return base64_imgs


@sefi_component.util(util_type=SubComponentType.MODEL_INFERENCE_DETECTOR, util_target=ComponentType.MODEL, name="HUAWEI(API)")
def inference_detector(model, img):
    results = []
    for single_img in img:
        results.append(model(single_img))
    return results


@sefi_component.util(util_type=SubComponentType.RESULT_POSTPROCESSOR, util_target=ComponentType.MODEL, name="HUAWEI(API)")
def result_post_handler(result, args):
    predicts = []
    results = []
    for single_result in result:
        score = {}
        for data in single_result.result.tags:
            score[data.tag] = data.confidence
        results.append(score)
        predicts.append(single_result.result.tags[0].tag)
    return predicts, results
