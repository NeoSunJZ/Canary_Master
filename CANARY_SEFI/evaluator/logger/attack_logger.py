from colorama import Fore
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_attack_log(atk_name, base_model, atk_type=None, atk_perturbation_budget=None):
    sql_insert = " INSERT INTO attack_log (attack_id, batch_id, atk_name, base_model, atk_type, atk_perturbation_budget) " + \
                 " VALUES (NULL,?,?,?,?,?)"
    sql_query = " SELECT attack_id FROM attack_log " \
                " WHERE batch_id = ? AND atk_name = ? AND base_model = ? AND atk_type = ? AND atk_perturbation_budget = ? "
    args = (str(batch_flag.batch_id), str(atk_name), str(base_model), str(atk_type), str(atk_perturbation_budget))
    result = log.query_log(sql_query, args)
    if result is not None:
        attack_id = result["attack_id"]
        if log.debug_log:
            msg = "[ LOGGER ] 日志存在 本次选定的攻击方法attack_id为 {}".format(attack_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return attack_id
    else:
        attack_id = log.insert_log(sql_insert,args)
        if log.debug_log:
            msg = "[ LOGGER ] 已写入日志 本次选定的攻击方法attack_id为 {}".format(attack_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return attack_id


def find_attack_log_by_name(batch_id, atk_name):
    sql = " SELECT * FROM attack_log WHERE batch_id = ? AND atk_name = ? "
    return log.query_logs(sql, (str(batch_id), str(atk_name)))


def find_attack_log_by_name_and_base_model(batch_id, atk_name, base_model, explore_perturbation_mode=False):
    sql = " SELECT * FROM attack_log WHERE batch_id = ? AND atk_name = ? AND base_model = ? "
    logs = log.query_logs(sql, (str(batch_id), str(atk_name), str(base_model)))
    if not explore_perturbation_mode:
        if len(logs) > 1:
            raise RuntimeError("more than one attack record found")
        return logs[0]
    else:
        return logs


def find_attack_log(attack_id):
    sql = " SELECT * FROM attack_log WHERE attack_id = ?"
    return log.query_log(sql, (attack_id,))


