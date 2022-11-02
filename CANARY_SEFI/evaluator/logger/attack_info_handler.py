from colorama import Fore
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


def add_attack_log(atk_name, base_model, atk_type=None, atk_perturbation_budget=None):
    sql_insert = " INSERT INTO attack_info_log (attack_id, atk_name, base_model, atk_type, atk_perturbation_budget) " + \
                 " VALUES (NULL,?,?,?,?)"
    sql_query = " SELECT attack_id FROM attack_info_log " \
                " WHERE atk_name = ? AND base_model = ? AND atk_type = ? AND atk_perturbation_budget = ? "
    args = (str(atk_name), str(base_model), str(atk_type), atk_perturbation_budget)
    result = task_manager.test_data_logger.query_log(sql_query, args)
    if result is not None:
        attack_id = result["attack_id"]
    else:
        attack_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Attack ID:{}. Attack Info:[Name:{} Base:{} Type:{} PerturbationBudget:{}]"\
            .format(attack_id, *args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return attack_id


def find_attack_log_by_name(atk_name):
    sql = " SELECT * FROM attack_info_log WHERE atk_name = ? "
    return task_manager.test_data_logger.query_logs(sql, (str(atk_name),))


def find_attack_log_by_name_and_base_model(atk_name, base_model, perturbation_increment_mode=False):
    sql = " SELECT * FROM attack_info_log WHERE atk_name = ? AND base_model = ? "
    logs = task_manager.test_data_logger.query_logs(sql, (str(atk_name), str(base_model)))

    if not perturbation_increment_mode:
        if len(logs) > 1:
            raise RuntimeError("[ Logic Error ] More than one attack record found!")
        return logs[0]
    else:
        return logs


def find_attack_log(attack_id):
    sql = " SELECT * FROM attack_info_log WHERE attack_id = ?"
    return task_manager.test_data_logger.query_log(sql, (attack_id,))


