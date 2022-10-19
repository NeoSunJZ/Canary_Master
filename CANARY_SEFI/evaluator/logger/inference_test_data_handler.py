import numpy as np
from colorama import Fore
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.entity.dataset_info_entity import DatasetType


def save_inference_test_data(img_id, img_type, inference_model, inference_img_label, inference_img_conf_array):
    sql = "INSERT INTO inference_test_data (inference_test_data_id, img_id, img_type, inference_model, inference_img_label, inference_img_conf_array) " \
          "VALUES (NULL,?,?,?,?,?)"
    args = (img_id, str(img_type), str(inference_model), str(inference_img_label), str(inference_img_conf_array))
    inference_test_data_id = batch_manager.test_data_logger.insert_log(sql, args)
    if batch_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Image(ImgID:{} Type:{}) has Inferenced by Model({}). Inference result is {}. ".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return inference_test_data_id


def read_conf_array(inference_img_conf_array):
    return np.array(eval(', '.join(inference_img_conf_array.split())))


def handle_result(inference_logs):
    for inference_log in inference_logs:
        inference_log["inference_img_conf_array"] = read_conf_array(inference_log["inference_img_conf_array"]).tolist()
    return inference_logs


def get_all_inference_test_data(img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? "
    return handle_result(batch_manager.test_data_logger.query_logs(sql, (str(img_type), )))


def get_inference_test_data_by_model_name(model_name, img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? AND inference_model = ?"
    return handle_result(batch_manager.test_data_logger.query_logs(sql, (str(img_type), str(model_name))))


def get_inference_test_data_by_img_id(img_id, img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? AND img_id = ?"
    return handle_result(batch_manager.test_data_logger.query_logs(sql, (str(img_type), img_id)))


def get_clean_inference_test_data_with_img_info(inference_model):
    sql = "SELECT inference_test_data.*, ori_img_log.ori_img_label FROM inference_test_data, ori_img_log " \
          "WHERE inference_test_data.img_type = 'NORMAL' AND inference_test_data.img_id = ori_img_log.ori_img_id " \
          "AND inference_test_data.inference_model = ? "
    return handle_result(batch_manager.test_data_logger.query_logs(sql, (str(inference_model),)))


def get_adv_inference_test_data_with_adv_info(inference_model, adv_example_file_type=DatasetType.ADVERSARIAL_EXAMPLE_IMG.value):
    sql = "SELECT inference_test_data.*, ori_img_log.ori_img_label, ori_img_log.ori_img_id, " \
          "attack_info_log.atk_name, attack_info_log.base_model " \
          "FROM inference_test_data, adv_img_file_log, ori_img_log, attack_info_log " \
          "WHERE inference_test_data.img_type = ? " \
          "AND inference_test_data.img_id = adv_img_file_log.adv_img_file_id " \
          "AND adv_img_file_log.ori_img_id = ori_img_log.ori_img_id " \
          "AND adv_img_file_log.attack_id = attack_info_log.attack_id " \
          "AND inference_test_data.inference_model = ? "
    return handle_result(batch_manager.test_data_logger.query_logs(sql, (str(adv_example_file_type),str(inference_model))))
