from flask import Blueprint, request

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.basic.dataset_function import dataset_image_reader, dataset_single_image_reader
from CANARY_SEFI.core.function.init_dataset import dataset_seed_handler
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo
from CANARY_SEFI.entity.msg_entity import MsgEntity
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_model_security_synthetical_capability_log, \
    get_explore_perturbation_result_by_attack_name_and_base_model
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import get_clean_inference_test_data_with_img_info, \
    get_adv_inference_test_data_with_adv_info
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_dataset, get_pic_base64_from_nparray
from CANARY_SEFI.handler.image_handler.img_utils import show_img_diff
from CANARY_SEFI.handler.json_handler.json_io_handler import get_info_from_json_file

api = Blueprint('analyzer_api', __name__)


@api.route('/result/getInferenceResultByModelName', methods=['GET'])
def get_inference_result_by_model_name():
    # 初始化批次
    batch_manager.init_batch(request.args.get("batchToken"))

    inference_model = request.args.get("inferenceModel")
    result = {
        "clean": get_clean_inference_test_data_with_img_info(inference_model),
        "adv": get_adv_inference_test_data_with_adv_info(inference_model),
    }
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getModelSecuritySyntheticalCapabilityResult', methods=['GET'])
def get_model_security_synthetical_capability_result():
    # 初始化批次
    batch_manager.init_batch(request.args.get("batchToken"))

    inference_model = request.args.get("inferenceModel")
    result = get_model_security_synthetical_capability_log(inference_model)
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getExplorePerturbationResult', methods=['GET'])
def get_explore_perturbation_result():
    # 初始化批次
    batch_manager.init_batch(request.args.get("batchToken"))

    atk_name = request.args.get("attackName")
    base_model = request.args.get("baseModel")
    result = get_explore_perturbation_result_by_attack_name_and_base_model(atk_name, base_model)
    return MsgEntity("SUCCESS", "1", result).msg2json()


@api.route('/result/getAdvInfo', methods=['GET'])
def get_adv_info_by_adv_img_id():
    # 初始化批次
    batch_manager.init_batch(request.args.get("batchToken"))

    adv_img_file_id = request.args.get("advImgId")
    need_adv_img = request.args.get("needAdvImg", 0)

    adv_example_file_log = find_adv_example_file_log_by_id(adv_img_file_id)

    if need_adv_img == "1":
        ori_img_log = find_img_log_by_id(adv_example_file_log["ori_img_id"])

        config = get_info_from_json_file("config.json")

        dataset_info = DatasetInfo(config.get('dataset'), int(dataset_seed_handler(config.get('dataset_seed',None))),
                                   int(config.get('dataset_size')))
        original_img = dataset_single_image_reader(dataset_info, int(ori_img_log["ori_img_cursor"]))

        adv_file_path = batch_manager.base_temp_path + "pic/"
        adversarial_img = get_pic_nparray_from_dataset(adv_file_path, adv_example_file_log["adv_img_filename"])

        adv_example_file_log['adv_img'] = {
            "original_img": get_pic_base64_from_nparray(original_img),
            "adversarial_img": get_pic_base64_from_nparray(adversarial_img),
            "diff": show_img_diff(original_img, adversarial_img)
        }

    return MsgEntity("SUCCESS", "1", adv_example_file_log).msg2json()
