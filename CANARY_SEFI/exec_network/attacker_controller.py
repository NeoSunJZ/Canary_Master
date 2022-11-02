import json
from flask import Blueprint, request
from flask_socketio import emit


from CANARY_SEFI.handler.image_handler.img_utils import show_img_diff
from CANARY_SEFI.handler.image_handler.img_io_handler import get_nparray_from_file_input, get_pic_base64_from_nparray, \
    get_temp_download_url
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.core.function.core.attacker_function import adv_attack_4_img, adv_attack_4_img_batch

api = Blueprint('attacker_api', __name__)


@api.route('/advExample/attackSingleImg', methods=['POST'])
def adv_attack_4_single_img():
    ori_img = get_nparray_from_file_input(request.files.get('img'))
    atk_name = request.form.get('atkName')
    atk_args = request.form.get('atkArgs')
    model_name = request.form.get('modelName')
    model_args = request.form.get('modelArgs')
    img_proc_args = request.form.get('imgProcArgs')
    adv_result = adv_attack_4_img(atk_name, atk_args, model_name, model_args, ori_img, img_proc_args)
    return get_pic_base64_from_nparray(adv_result)


@api.route('/advExample/attackBatchImg', methods=['POST'])
def attack_task_img():
    atk_name = request.form.get('atkName')
    atk_args = request.form.get('atkArgs')
    model_name = request.form.get('modelName')
    model_args = request.form.get('modelArgs')
    img_proc_args = request.form.get('imgProcArgs')

    img_list = json.loads(request.form.get('imgList'))
    dataset_name = request.form.get('datasetName')

    def each_img_finish_callback(img, adv_result):
        if request.form.get('room') is not None:
            emit("attack_result", show_img_diff(img, adv_result), room=request.form.get('room'), namespace='/websocket')

    task_token = adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_name,
                                         img_list, each_img_finish_callback)
    if request.form.get('returnType') == 'token':
        return MsgEntity("SUCCESS", "1", task_token).msg2json()
    else:
        return get_temp_download_url(task_token)
