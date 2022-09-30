from flask import Blueprint, request

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader, dataset_single_image_reader
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_log_by_img_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_model_test_result_log, \
    get_model_security_synthetical_capability_log, get_adv_da_test_result_log_by_attack_name, \
    get_attack_test_result_log_by_attack_name, get_explore_perturbation_result_by_attack_name_and_base_model
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log, find_dataset_log
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import \
    find_adv_inference_log_with_img_info, find_clean_inference_log_with_img_info
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_dataset, get_pic_base64_from_nparray
from CANARY_SEFI.handler.image_handler.img_utils import show_img_diff

api = Blueprint('analyzer_api', __name__)


@api.route('/result/getInferenceResultByModelName', methods=['GET'])
def get_inference_result_by_model_name():
    inference_model = request.args.get("inferenceModel")
    batch_id = request.args.get("batchToken")
    result = {
        "clean": find_clean_inference_log_with_img_info(batch_id, inference_model),
        "adv": find_adv_inference_log_with_img_info(batch_id, inference_model),
    }
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getModelSecuritySyntheticalCapabilityResult', methods=['GET'])
def get_model_security_synthetical_capability_result():
    inference_model = request.args.get("inferenceModel")
    batch_id = request.args.get("batchToken")
    result = get_model_security_synthetical_capability_log(batch_id, inference_model)
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getExplorePerturbationResult', methods=['GET'])
def get_explore_perturbation_result():
    atk_name = request.args.get("attackName")
    base_model = request.args.get("baseModel")
    batch_id = request.args.get("batchToken")
    result = get_explore_perturbation_result_by_attack_name_and_base_model(batch_id, atk_name, base_model)
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getAdvInfo', methods=['GET'])
def get_adv_info_by_adv_img_id():
    adv_img_id = request.args.get("advImgId")
    need_adv_img = request.args.get("needAdvImg", 0)

    adv_example_log = find_adv_log_by_img_id(adv_img_id)

    if need_adv_img == "1":
        ori_img_log = find_img_log(adv_example_log["ori_img_id"])

        dataset_log = find_dataset_log(ori_img_log["dataset_id"])
        dataset_info = DatasetInfo(dataset_log['dataset_name'], int(dataset_log['dataset_seed']),
                                   int(dataset_log['dataset_size']))

        original_img = dataset_single_image_reader(dataset_info, int(ori_img_log["ori_img_cursor"]))

        adv_dataset_temp_path = config_manager.config.get("temp", "Dataset_Temp/")
        adv_file_path = adv_dataset_temp_path + adv_example_log["batch_id"] + "/"
        adversarial_img = get_pic_nparray_from_dataset(adv_file_path, adv_example_log["adv_img_filename"])

        adv_example_log['adv_img'] = {
            "original_img": get_pic_base64_from_nparray(original_img),
            "adversarial_img": get_pic_base64_from_nparray(adversarial_img),
            "diff": show_img_diff(original_img, adversarial_img)
        }

    return MsgEntity("SUCCESS", "1", adv_example_log).msg2json()
