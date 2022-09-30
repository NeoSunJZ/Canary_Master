from colorama import Fore
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

logger = batch_manager.test_data_logger


# 新增对抗样本文件记录
def add_adv_example_file_log(attack_id, ori_img_id, adv_img_filename, adv_raw_nparray_filename):
    sql = "INSERT INTO adv_img_file_log (adv_img_file_id, attack_id, ori_img_id, adv_img_filename, adv_raw_nparray_filename) " \
          "VALUES (NULL,?,?,?,?)"
    adv_img_file_id = logger.insert_log(sql, (attack_id, ori_img_id, str(adv_img_filename), str(adv_raw_nparray_filename)))
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Img file name: {}. Numpy array file name: {}."\
            .format(adv_img_file_id, adv_img_filename, adv_raw_nparray_filename)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return adv_img_file_id


# 设置对抗样本文件生成耗时
def set_adv_example_file_cost_time(adv_img_file_id, cost_time):
    sql = "UPDATE adv_img_file_log SET cost_time = ? WHERE adv_img_file_id = ?"
    logger.update_log(sql, (cost_time, adv_img_file_id))
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Cost Time: {}.".format(adv_img_file_id, cost_time)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


def find_all_adv_example_file_logs():
    sql = "SELECT * FROM adv_img_file_log"
    return logger.query_logs(sql, ())


def find_adv_example_file_logs_by_ori_img_id(ori_img_id):
    sql = "SELECT * FROM adv_img_file_log WHERE ori_img_id = ?"
    return logger.query_logs(sql, (ori_img_id,))


def find_adv_example_file_logs_by_attack_id(attack_id):
    sql = "SELECT * FROM adv_img_file_log WHERE attack_id = ?"
    return logger.query_logs(sql, (attack_id,))


def find_adv_example_file_log_by_id(adv_img_file_id):
    sql = " SELECT * FROM adv_img_file_log WHERE adv_img_file_id = ? "
    return logger.query_log(sql, (adv_img_file_id,))
