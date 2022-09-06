from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_dataset_log(dataset_name, dataset_seed, dataset_size):
    sql = " INSERT INTO dataset_log (dataset_id,batch_id,dataset_name,dataset_seed,dataset_size) " + \
          " VALUES (NULL,'{}','{}','{}','{}')" \
              .format(str(batch_flag.batch_id), str(dataset_name), str(dataset_seed), str(dataset_size))
    dataset_id = log.insert_log(sql)
    if log.debug_log:
        msg = "[ LOGGER ] 已写入日志 本次测试Dataset_id为{}".format(dataset_id)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return dataset_id


def find_dataset_log(dataset_id):
    sql = " SELECT * FROM dataset_log WHERE dataset_id = '{}' AND batch_id = '{}'"\
        .format(str(dataset_id), str(batch_flag.batch_id))
    return log.query_log(sql)


def add_img_log(dataset_id, ori_img_label, ori_img_cursor):
    sql_insert = " INSERT INTO ori_img_log (ori_img_id,batch_id,dataset_id,ori_img_label,ori_img_cursor) " + \
                 " VALUES (NULL,'{}','{}','{}','{}')" \
                     .format(str(batch_flag.batch_id), str(dataset_id), str(ori_img_label), str(ori_img_cursor))

    sql_query = " SELECT ori_img_id FROM ori_img_log " \
                " WHERE dataset_id = '{}' AND batch_id = '{}' AND ori_img_label = '{}' AND ori_img_cursor = '{}'"\
        .format(str(dataset_id), str(batch_flag.batch_id), str(ori_img_label), str(ori_img_cursor))
    result = log.query_log(sql_query)
    if len(result) != 0:
        img_id = result[0][0]
        if log.debug_log:
            msg = "[ LOGGER ] 日志存在 本次选定的图片img_id为{}".format(img_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return img_id
    else:
        img_id = log.insert_log(sql_insert)
        if log.debug_log:
            msg = "[ LOGGER ] 已写入日志 本次选定的图片img_id为{}".format(img_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return img_id

def find_img_log(ori_img_id):
    sql = " SELECT * FROM ori_img_log WHERE ori_img_id = '{}'".format(str(ori_img_id))
    return log.query_log(sql)
