from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


def add_trans_log(attack_id, trans_name):
    sql_insert = " INSERT INTO trans_info_log (trans_id, attack_id, trans_name) " + \
                 " VALUES (NULL,?,?)"
    sql_query = " SELECT trans_id FROM trans_info_log " \
                " WHERE attack_id = ? AND trans_name = ? "
    args = (attack_id, str(trans_name))
    result = task_manager.test_data_logger.query_log(sql_query, args)
    if result is not None:
        trans_id = result["trans_id"]
    else:
        trans_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Trans ID:{}. Trans Info:[Attack ID:{} Trans Name:{}]" \
            .format(trans_id, *args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return trans_id


def find_trans_log_by_name(trans_name):
    sql = " SELECT * FROM trans_info_log WHERE trans_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (str(trans_name),))


def find_trans_log_by_name_and_atk_id(trans_name, attack_id):
    sql = " SELECT * FROM trans_info_log WHERE trans_name = ? AND attack_id = ? "
    logs = task_manager.test_data_logger.query_logs(sql, (str(trans_name), str(attack_id)))

    return logs


def find_trans_log(trans_id):
    sql = " SELECT * FROM trans_info_log WHERE trans_id = ?"
    return task_manager.test_data_logger.query_log(sql, (trans_id,))
