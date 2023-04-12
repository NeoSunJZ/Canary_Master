from colorama import Fore
from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.helper.realtime_reporter import reporter


# 存储对抗样本文件扰动评估数据(可覆盖之前的结果)
def save_adv_example_da_test_data(adv_img_file_id, adv_example_file_type, adv_da_test_result):
    sql = "REPLACE INTO adv_example_da_test_data (adv_img_file_id, adv_example_file_type, " \
          "maximum_disturbance, euclidean_distortion, high_freq_euclidean_distortion, low_freq_euclidean_distortion," \
          " pixel_change_ratio, deep_metrics_similarity, low_level_metrics_similarity) VALUES (?,?,?,?,?,?,?,?,?)"
    args = (adv_img_file_id, adv_example_file_type,
            adv_da_test_result.get("maximum_disturbance"), adv_da_test_result.get("euclidean_distortion"),
            adv_da_test_result.get("high_freq_euclidean_distortion"),
            adv_da_test_result.get("low_freq_euclidean_distortion"),
            adv_da_test_result.get("pixel_change_ratio"),
            adv_da_test_result.get("deep_metrics_similarity"),
            adv_da_test_result.get("low_level_metrics_similarity"))
    task_manager.test_data_logger.insert_log(sql, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {} ({}). " \
              "Adversarial-example Disturbance Evaluation Result :" \
              "[MD(L-inf):{}, ED(L2):{}, ED-HF(L2):{}, ED-LF(L2):{}, PCR(L0):{}, DMS(DISTS):{}, LMS(MS-GMSD):{}]."\
            .format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


def find_adv_example_da_test_data_by_id_and_type(adv_img_file_id, adv_example_file_type):
    sql = "SELECT * FROM adv_example_da_test_data WHERE adv_img_file_id = ? AND adv_example_file_type = ? "
    return task_manager.test_data_logger.query_log(sql, (adv_img_file_id,adv_example_file_type))