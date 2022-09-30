from colorama import Fore
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

logger = batch_manager.test_data_logger


def add_attack_log(atk_name, base_model, atk_type=None, atk_perturbation_budget=None):
    sql_insert = " INSERT INTO attack_info_log (attack_id, atk_name, base_model, atk_type, atk_perturbation_budget) " + \
                 " VALUES (NULL,?,?,?,?)"
    sql_query = " SELECT attack_id FROM attack_info_log " \
                " WHERE atk_name = ? AND base_model = ? AND atk_type = ? AND atk_perturbation_budget = ? "
    args = (str(atk_name), str(base_model), str(atk_type), str(atk_perturbation_budget))
    result = logger.query_log(sql_query, args)
    if result is not None:
        attack_id = result["attack_id"]
    else:
        attack_id = logger.insert_log(sql_insert, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Attack ID:{}. Attack Info:[Name:{} Base:{} Type:{} PerturbationBudget:{}]"\
            .format(attack_id, *args)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return attack_id


def find_attack_log_by_name(atk_name):
    sql = " SELECT * FROM attack_info_log WHERE atk_name = ? "
    return logger.query_logs(sql, (str(atk_name),))


def find_attack_log_by_name_and_base_model(atk_name, base_model, explore_perturbation_mode=False):
    sql = " SELECT * FROM attack_info_log WHERE atk_name = ? AND base_model = ? "
    logs = logger.query_logs(sql, (str(atk_name), str(base_model)))

    if not explore_perturbation_mode:
        if len(logs) > 1:
            raise RuntimeError("[ Logic Error ] More than one attack record found!")
        return logs[0]
    else:
        return logs


def find_attack_log(attack_id):
    sql = " SELECT * FROM attack_info_log WHERE attack_id = ?"
    return logger.query_log(sql, (attack_id,))


