from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


# 新增对抗样本文件记录
def add_adv_example_file_log(attack_id, ori_img_id, adv_img_filename, adv_raw_nparray_filename, tlabel=None):
    sql_insert = "INSERT INTO adv_img_file_log (adv_img_file_id, attack_id, ori_img_id, adv_img_filename, adv_raw_nparray_filename, tlabel) " \
          "VALUES (NULL,?,?,?,?,?)"

    sql_query = " SELECT adv_img_file_id FROM adv_img_file_log " \
                " WHERE attack_id = ? AND ori_img_id = ? AND adv_img_filename = ? AND adv_raw_nparray_filename = ? AND tlabel = ?"
    args = (attack_id, ori_img_id, str(adv_img_filename), str(adv_raw_nparray_filename), str(tlabel))
    result = task_manager.test_data_logger.query_log(sql_query, args)

    if result is not None:
        adv_img_file_id = result["adv_img_file_id"]
    else:
        adv_img_file_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Img file name: {}. Numpy array file name: {}."\
            .format(adv_img_file_id, adv_img_filename, adv_raw_nparray_filename)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return adv_img_file_id


# 设置对抗样本文件生成耗时
def set_adv_example_file_cost_time(adv_img_file_id, cost_time):
    sql = "UPDATE adv_img_file_log SET cost_time = ? WHERE adv_img_file_id = ?"
    task_manager.test_data_logger.update_log(sql, (cost_time, adv_img_file_id))
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Cost Time: {}.".format(adv_img_file_id, cost_time)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


# 设置对抗样本文件生成模型查询量
def set_adv_example_file_query_num(adv_img_file_id, query_num):
    query_num_forward, query_num_backward = query_num.get("forward"), query_num.get("backward")
    sql = "UPDATE adv_img_file_log SET query_num_forward = ?,query_num_backward = ? WHERE adv_img_file_id = ?"
    task_manager.test_data_logger.update_log(sql, (query_num_forward, query_num_backward, adv_img_file_id))
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Query Num (Forward): {} , Query Num (Backward): {}."\
            .format(adv_img_file_id, query_num_forward, query_num_backward)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


# 设置对抗样本文件有效性
def set_adv_example_file_ground_valid(adv_img_file_id, ground_valid):
    sql = "UPDATE adv_img_file_log SET ground_valid = ? WHERE adv_img_file_id = ?"
    task_manager.test_data_logger.update_log(sql, (ground_valid, adv_img_file_id))
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-example file ID: {}. Ground Valid: {}.".format(adv_img_file_id, ground_valid)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


def find_all_adv_example_file_logs():
    sql = "SELECT * FROM adv_img_file_log"
    return task_manager.test_data_logger.query_logs(sql, ())


def find_adv_example_file_logs_by_ori_img_id(ori_img_id):
    sql = "SELECT * FROM adv_img_file_log WHERE ori_img_id = ?"
    return task_manager.test_data_logger.query_logs(sql, (ori_img_id,))


def find_adv_example_file_logs_by_attack_id(attack_id):
    sql = "SELECT * FROM adv_img_file_log WHERE attack_id = ?"
    return task_manager.test_data_logger.query_logs(sql, (attack_id,))


def find_adv_example_file_log_by_id(adv_img_file_id):
    sql = " SELECT * FROM adv_img_file_log WHERE adv_img_file_id = ? "
    return task_manager.test_data_logger.query_log(sql, (adv_img_file_id,))
