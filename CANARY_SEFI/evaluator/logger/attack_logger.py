from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_attack_log(atk_name, base_model, atk_type=None):
    sql_insert = " INSERT INTO attack_log (attack_id,batch_id,atk_name,base_model,atk_type) " + \
                 " VALUES (NULL,'{}','{}','{}','{}')" \
                     .format(str(batch_flag.batch_id), str(atk_name), str(base_model), str(atk_type))

    sql_query = " SELECT attack_id FROM attack_log " \
                " WHERE atk_name = '{}' AND batch_id = '{}' AND base_model = '{}' AND atk_type = '{}'"\
        .format(str(atk_name), str(batch_flag.batch_id), str(base_model), str(atk_type))
    result = log.query_log(sql_query)

    if len(result) != 0:
        attack_id = result[0][0]
        if log.debug_log:
            tqdm.write(Fore.CYAN + "[LOGGER] 日志存在 本次选定的攻击方法attack_id为{}".format(attack_id))
            tqdm.write(Style.RESET_ALL)
        return attack_id
    else:
        attack_id = log.insert_log(sql_insert)
        if log.debug_log:
            tqdm.write(Fore.CYAN + "[LOGGER] 已写入日志 本次选定的攻击方法attack_id为{}".format(attack_id))
            tqdm.write(Style.RESET_ALL)
        return attack_id

def find_attack_log(attack_id):
    sql = " SELECT * FROM attack_log WHERE attack_id = '{}'".format(str(attack_id))
    return log.query_log(sql)

