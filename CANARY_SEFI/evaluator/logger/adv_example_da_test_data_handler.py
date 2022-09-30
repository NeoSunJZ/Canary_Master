from colorama import Fore
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

logger = batch_manager.test_data_logger


# 存储对抗样本文件扰动评估数据(可覆盖之前的结果)
def save_adv_example_da_test_data(adv_img_file_id, adv_example_file_type, adv_da_test_result):
    sql = "REPLACE INTO adv_example_da_test_data (adv_img_file_id, adv_example_file_type, " \
          "maximum_disturbance, euclidean_distortion, pixel_change_ratio, " \
          "deep_metrics_similarity, low_level_metrics_similarity) VALUES (?,?,?,?,?,?,?)"
    args = (adv_img_file_id, adv_example_file_type,
            adv_da_test_result.get("maximum_disturbance"), adv_da_test_result.get("euclidean_distortion"),
            adv_da_test_result.get("pixel_change_ratio"),
            adv_da_test_result.get("deep_metrics_similarity"),
            adv_da_test_result.get("low_level_metrics_similarity"))
    logger.insert_log(sql, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {} ({}). " \
              "Adversarial-example Disturbance Evaluation Result :" \
              "[MD(L-inf):{}, ED(L2):{}, PCR(L0):{}, DMS(DISTS):{}, LMS(MS-GMSD):{}]."\
            .format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
