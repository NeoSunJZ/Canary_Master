from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_adv_base_log(attack_id, ori_img_id, adv_img_filename):
    sql = "INSERT INTO adv_example_log (adv_img_id,batch_id,attack_id,ori_img_id,adv_img_filename) " \
          "VALUES (NULL, '{}', '{}', '{}', '{}')" \
        .format(str(batch_flag.batch_id), str(attack_id), str(ori_img_id), str(adv_img_filename))

    adv_img_id = log.insert_log(sql)
    if log.debug_log:
        tqdm.write(Fore.CYAN + "[LOGGER] 已写入日志 本次攻击生成adv_img_id为 {} 存储文件名{}".format(adv_img_id, adv_img_filename))
        tqdm.write(Style.RESET_ALL)
    return adv_img_id


def add_adv_build_log(adv_img_id, cost_time):
    sql = "UPDATE adv_example_log SET cost_time = '{}' WHERE adv_img_id = '{}'".format(str(cost_time), str(adv_img_id))
    log.update_log(sql)
    if log.debug_log:
        tqdm.write(Fore.CYAN + "[LOGGER] 已写入日志 adv_img_id为 {} 的对抗样本耗时{}".format(adv_img_id, cost_time))


def add_adv_da_log(adv_img_id, adv_da_test_result):
    sql = "UPDATE adv_example_log " \
          "SET adv_img_maximum_disturbance = '{}',adv_img_euclidean_distortion = '{}',adv_img_pixel_change_ratio = '{}'," \
          "adv_img_deep_metrics_similarity = '{}',adv_img_low_level_metrics_similarity = '{}' " \
          "WHERE adv_img_id = '{}'" \
        .format(str(adv_da_test_result.get("maximum_disturbance")), str(adv_da_test_result.get("euclidean_distortion")),
                str(adv_da_test_result.get("pixel_change_ratio")), str(adv_da_test_result.get("deep_metrics_similarity")),
                str(adv_da_test_result.get("low_level_metrics_similarity")), str(adv_img_id))
    log.update_log(sql)
    if log.debug_log:
        tqdm.write(Fore.CYAN + "[LOGGER] 已写入日志 adv_img_id为{}的对抗样本 "
                          "扰动感知评估： MD(L-inf) {}, ED(L2) {}, PCR(L0) {}, DMS(DISTS) {}, LMS(MS-GMSD) {}"
              .format(adv_img_id, adv_da_test_result.get("maximum_disturbance"), adv_da_test_result.get("euclidean_distortion"),
                      adv_da_test_result.get("pixel_change_ratio"), adv_da_test_result.get("deep_metrics_similarity"),
                      adv_da_test_result.get("low_level_metrics_similarity")))


def find_batch_adv_log(batch_id):
    sql = " SELECT * FROM adv_example_log WHERE batch_id = '{}'".format(str(batch_id))
    return log.query_log(sql)


def find_adv_log(adv_img_id):
    sql = " SELECT * FROM adv_example_log WHERE adv_img_id = '{}'".format(str(adv_img_id))
    return log.query_log(sql)
