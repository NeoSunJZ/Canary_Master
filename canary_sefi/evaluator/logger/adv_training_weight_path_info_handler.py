from colorama import Fore
from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.helper.realtime_reporter import reporter


def add_adv_training_weight_file_path_log(model_name, defense_name, epoch_cursor, weight_file_path):
    sql = "REPLACE INTO adv_training_weight_path_info (model_name, defense_name, epoch_cursor, weight_file_path) " \
          "VALUES (?,?,?,?)"

    args = (model_name, defense_name, epoch_cursor, weight_file_path)
    task_manager.test_data_logger.query_log(sql, args)

    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Adversarial-training ( method {} on model {} ) weight path ( epoch : {} ) : {}".format(*args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")

def get_adv_training_weight_file_path(model_name, defense_name, epoch_cursor):
    sql = "SELECT weight_file_path FROM adv_training_weight_path_info WHERE model_name = ? AND defense_name = ? AND epoch_cursor = ?"
    path_dic = task_manager.test_data_logger.query_log(sql, (model_name, defense_name, epoch_cursor))
    file_path = path_dic['weight_file_path']
    return file_path

def get_all_AT_weight_file_path():
    sql = "SELECT * FROM adv_training_weight_path_info "
    path_list = task_manager.test_data_logger.query_logs(sql, ())
    return path_list


