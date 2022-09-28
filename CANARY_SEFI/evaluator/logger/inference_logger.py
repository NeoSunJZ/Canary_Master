import numpy as np
from colorama import Fore
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_inference_log(img_id, img_type, inference_model, inference_img_label, inference_img_conf_array):
    sql = "INSERT INTO inference_result (inference_result_id, batch_id, img_id, img_type, inference_model, inference_img_label, inference_img_conf_array) " \
          "VALUES (NULL,?,?,?,?,?,?)"

    inference_result_id = \
        log.insert_log(sql, (str(batch_flag.batch_id), str(img_id), str(img_type), str(inference_model),
                             str(inference_img_label), str(inference_img_conf_array)))
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 推断结果inference_result_id为 {} (基于 {} 推理图片(img_id {})的标签为 {})"\
            .format(inference_result_id, inference_model, img_id, inference_img_label)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")

    return img_id


def read_conf_array(inference_img_conf_array):
    return np.array(eval(', '.join(inference_img_conf_array.split())))


def handle_result(inference_logs):
    for inference_log in inference_logs:
        inference_log["inference_img_conf_array"] = np.array(eval(', '.join(inference_log["inference_img_conf_array"].split()))).tolist()
    return inference_logs


def find_all_inference_log(img_type, batch_id):
    sql = " SELECT * FROM inference_result WHERE img_type = ? AND batch_id = ?"
    return handle_result(log.query_logs(sql, (str(img_type), str(batch_id))))


def find_inference_log_by_model_name(img_type, batch_id, model_name):
    sql = " SELECT * FROM inference_result WHERE img_type = ? AND batch_id = ? AND inference_model = ?"
    return handle_result(log.query_logs(sql, (str(img_type), str(batch_id), str(model_name))))


def find_inference_log_by_img_id(img_type, img_id):
    sql = " SELECT * FROM inference_result WHERE img_type = ? AND img_id = ?"
    return handle_result(log.query_logs(sql, (str(img_type), img_id)))


def find_clean_inference_log_with_img_info(batch_id, model_name):
    sql = " SELECT inference_result.*,ori_img_log.ori_img_label FROM inference_result, ori_img_log" \
          " WHERE inference_result.img_type = 'NORMAL' AND inference_result.img_id = ori_img_log.ori_img_id" \
          " AND inference_result.batch_id = ? AND inference_result.inference_model = ?"
    return handle_result(log.query_logs(sql, (str(batch_id), str(model_name))))

def find_adv_inference_log_with_img_info(batch_id, model_name):
    sql = " SELECT inference_result.*, ori_img_log.ori_img_label, ori_img_log.ori_img_id, attack_log.atk_name, attack_log.base_model " \
          " FROM inference_result, adv_example_log, ori_img_log, attack_log" \
          " WHERE inference_result.img_type = 'ADVERSARIAL_EXAMPLE' " \
          " AND inference_result.img_id = adv_example_log.adv_img_id " \
          " AND adv_example_log.ori_img_id = ori_img_log.ori_img_id" \
          " AND adv_example_log.attack_id = attack_log.attack_id" \
          " AND inference_result.batch_id = ? AND inference_result.inference_model = ?"
    return handle_result(log.query_logs(sql, (str(batch_id), str(model_name))))
