import json
from flask import Blueprint, request
from CANARY_SEFI.core.function.core.model_function import inference_detector_4_img, inference_detector_4_img_batch
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.handler.image_handler.img_io_handler import get_nparray_from_file_input

api = Blueprint('model_api', __name__)


@api.route('/inferenceDetector/inferenceImg', methods=['POST'])
def inference_detector_single_img():
    ori_img = get_nparray_from_file_input(request.files.get('img'))
    if request.files.get('img') is None:
        return MsgEntity("error", -1, "No Img find, please check IMG FILE").msg2json()

    model_name = request.form.get('modelName')
    model_args = request.form.get('modelArgs')
    img_proc_args = request.form.get('imgProcArgs')

    result = inference_detector_4_img(model_name, model_args, ori_img, img_proc_args)
    return result


@api.route('/inferenceDetector/inferenceBatchImg', methods=['POST'])
def inference_detector_batch_img():
    model_name = request.form.get('modelName')
    model_args = request.form.get('modelArgs')
    img_proc_args = request.form.get('imgProcArgs')

    img_list = json.loads(request.form.get('imgList'))
    dataset_name = request.form.get('datasetName')

    result = inference_detector_4_img_batch(model_name, model_args, img_proc_args, dataset_name, img_list, None)
    return result
