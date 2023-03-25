import pickle

import numpy as np
from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.entity.dataset_info_entity import DatasetType


def save_inference_test_data(img_id, img_type, inference_model, inference_img_label, inference_img_conf_array,
                             inference_cams=(None, None), use_pickle_dump=True):
    true_class_cams, inference_class_cams = inference_cams
    sql_query = "SELECT inference_test_data_id FROM  inference_test_data WHERE img_id = ? AND img_type = ? AND inference_model = ?"
    result = task_manager.test_data_logger.query_log(sql_query, (img_id, str(img_type), str(inference_model)))
    if result is not None:
        task_manager.test_data_logger.update_log("DELETE FROM inference_test_data WHERE inference_test_data_id = ?",
                                                 (result['inference_test_data_id'],))
    sql_insert = "INSERT INTO inference_test_data (inference_test_data_id, img_id, img_type, " \
          "inference_model, inference_img_label, inference_img_conf_array, true_class_cams, inference_class_cams) " \
          "VALUES (NULL,?,?,?,?,?,?,?)"
    if use_pickle_dump:
        inference_img_conf_array = pickle.dumps(inference_img_conf_array)
        if true_class_cams is not None:
            true_class_cams = pickle.dumps(true_class_cams)
        if inference_class_cams is not None:
            inference_class_cams = pickle.dumps(inference_class_cams)
    args = (img_id, str(img_type), str(inference_model),
            str(inference_img_label),
            str(inference_img_conf_array),
            str(true_class_cams),
            str(inference_class_cams))
    inference_test_data_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Image(ImgID:{} Type:{}) has Inferenced by Model({}). Inference result is {}. ".format(*args)
        if true_class_cams is not None and inference_class_cams is not None:
            msg += "G-CAM(Gradient-weighted Class Activation Mapping): Ready"
        else:
            msg += "G-CAM(Gradient-weighted Class Activation Mapping): Offline"
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return inference_test_data_id


def handle_result(inference_logs):
    for inference_log in inference_logs:
        inference_log["inference_img_conf_array"] = pickle.loads(eval(inference_log["inference_img_conf_array"]))
        inference_log["inference_class_cams"] = pickle.loads(eval(inference_log["inference_class_cams"]))
        inference_log["true_class_cams"] = pickle.loads(eval(inference_log["true_class_cams"]))
    return inference_logs


def get_all_inference_test_data(img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? "
    return handle_result(task_manager.test_data_logger.query_logs(sql, (str(img_type), )))


def get_inference_test_data_by_model_name(model_name, img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? AND inference_model = ?"
    return handle_result(task_manager.test_data_logger.query_logs(sql, (str(img_type), str(model_name))))


def get_inference_test_data_by_img_id(img_id, img_type):
    sql = "SELECT * FROM inference_test_data WHERE img_type = ? AND img_id = ?"
    return handle_result(task_manager.test_data_logger.query_logs(sql, (str(img_type), img_id)))


def get_clean_inference_test_data_with_img_info(inference_model):
    sql = "SELECT inference_test_data.*, ori_img_log.ori_img_label FROM inference_test_data, ori_img_log " \
          "WHERE inference_test_data.img_type = 'NORMAL' AND inference_test_data.img_id = ori_img_log.ori_img_id " \
          "AND inference_test_data.inference_model = ? "
    return handle_result(task_manager.test_data_logger.query_logs(sql, (str(inference_model),)))


def get_adv_inference_test_data_with_adv_info(inference_model, adv_example_file_type=DatasetType.ADVERSARIAL_EXAMPLE_IMG.value):
    sql = "SELECT inference_test_data.*, ori_img_log.ori_img_label, ori_img_log.ori_img_id, " \
          "attack_info_log.atk_name, attack_info_log.base_model " \
          "FROM inference_test_data, adv_img_file_log, ori_img_log, attack_info_log " \
          "WHERE inference_test_data.img_type = ? " \
          "AND inference_test_data.img_id = adv_img_file_log.adv_img_file_id " \
          "AND adv_img_file_log.ori_img_id = ori_img_log.ori_img_id " \
          "AND adv_img_file_log.attack_id = attack_info_log.attack_id " \
          "AND inference_test_data.inference_model = ? "
    return handle_result(task_manager.test_data_logger.query_logs(sql, (str(adv_example_file_type),str(inference_model))))
