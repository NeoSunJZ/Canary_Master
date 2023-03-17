from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


# 新增对抗样本转换文件记录
def add_adv_trans_img_file_log(trans_name, attack_id, adv_img_file_id, adv_trans_img_filename, ground_valid=None):
    sql_insert = "INSERT INTO adv_trans_img_file_log (adv_trans_img_file_id, trans_name, attack_id, adv_img_file_id, adv_trans_img_filename, ground_valid) " \
          "VALUES (NULL,?,?,?,?,?)"

    sql_query = " SELECT adv_trans_img_file_id FROM adv_trans_img_file_log " \
                " WHERE trans_name = ? AND attack_id = ? AND adv_img_file_id = ? AND adv_trans_img_filename = ? AND ground_valid = ?"
    args = (trans_name, attack_id, adv_img_file_id, adv_trans_img_filename, ground_valid)
    result = task_manager.test_data_logger.query_log(sql_query, args)

    if result is not None:
        adv_trans_img_file_id = result["adv_trans_img_file_id"]
    else:
        adv_trans_img_file_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adv_Trans file ID: {}. Trans name: {}. Trans Img file name: {}. Adv_Img file ID: {}."\
            .format(adv_trans_img_file_id, trans_name, adv_trans_img_filename, adv_img_file_id)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return adv_trans_img_file_id


# 设置对抗样本转换文件有效性
def set_adv_trans_file_ground_valid(adv_trans_img_file_id, ground_valid):
    sql = "UPDATE adv_trans_img_file_log SET ground_valid = ? WHERE adv_trans_img_file_id = ?"
    task_manager.test_data_logger.update_log(sql, (ground_valid, adv_trans_img_file_id))
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adv_trans file ID: {}. Ground Valid: {}.".format(adv_trans_img_file_id, ground_valid)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")


def find_all_adv_trans_file_logs():
    sql = "SELECT * FROM adv_trans_img_file_log"
    return task_manager.test_data_logger.query_logs(sql, ())


def find_adv_trans_file_logs_by_attack_id_and_trans_name(attack_id, trans_name):
    sql = "SELECT * FROM adv_trans_img_file_log WHERE attack_id = ? AND trans_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (attack_id, trans_name))


def find_adv_trans_file_log_by_id(adv_trans_img_file_id):
    sql = " SELECT * FROM adv_trans_img_file_log WHERE adv_trans_img_file_id = ? "
    return task_manager.test_data_logger.query_log(sql, (adv_trans_img_file_id,))
