from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_attack_log(atk_name, base_model, atk_type=None, atk_perturbation_budget=None):
    sql_insert = " INSERT INTO attack_log (attack_id,batch_id,atk_name,base_model,atk_type,atk_perturbation_budget) " + \
                 " VALUES (NULL,'{}','{}','{}','{}','{}')" \
                     .format(str(batch_flag.batch_id), str(atk_name), str(base_model), str(atk_type), str(atk_perturbation_budget))

    sql_query = " SELECT attack_id FROM attack_log " \
                " WHERE atk_name = '{}' AND batch_id = '{}' AND base_model = '{}' AND atk_type = '{}' AND atk_perturbation_budget = '{}'"\
        .format(str(atk_name), str(batch_flag.batch_id), str(base_model), str(atk_type), str(atk_perturbation_budget))
    result = log.query_log(sql_query)

    if len(result) != 0:
        attack_id = result[0][0]
        if log.debug_log:
            msg = "[ LOGGER ] 日志存在 本次选定的攻击方法attack_id为{}".format(attack_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return attack_id
    else:
        attack_id = log.insert_log(sql_insert)
        if log.debug_log:
            msg = "[ LOGGER ] 已写入日志 本次选定的攻击方法attack_id为{}".format(attack_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return attack_id

def find_attack_log(attack_id):
    sql = " SELECT * FROM attack_log WHERE attack_id = '{}'".format(str(attack_id))
    return log.query_log(sql)

