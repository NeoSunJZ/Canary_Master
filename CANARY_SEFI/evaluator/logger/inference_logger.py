from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_inference_log(img_id, img_type, inference_model, inference_img_label, inference_img_conf_array):
    sql = "INSERT INTO inference_result (inference_result_id,batch_id,img_id,img_type,inference_model,inference_img_label,inference_img_conf_array) " \
          "VALUES (NULL,'{}','{}','{}','{}','{}','{}')"\
        .format(str(batch_flag.batch_id),str(img_id),str(img_type),str(inference_model),str(inference_img_label),str(inference_img_conf_array))

    inference_result_id = log.insert_log(sql)
    if log.debug_log:
        tqdm.write(Fore.CYAN + "[LOGGER] 已写入日志  推断结果inference_result_id为{} (基于 {} 推理图片(img_id {})的标签为 {})".format(inference_result_id, inference_model, img_id, inference_img_label))
        tqdm.write(Style.RESET_ALL)
    return img_id


def find_all_inference_log(img_type, batch_id):
    sql = " SELECT * FROM inference_result " \
          " WHERE img_type = '{}' AND batch_id = '{}'".format(str(img_type), str(batch_id))
    return log.query_log(sql)

def find_inference_log(img_type, img_id):
    sql = " SELECT * FROM inference_result " \
          " WHERE img_type = '{}' AND img_id = '{}'".format(str(img_type), str(img_id))
    return log.query_log(sql)