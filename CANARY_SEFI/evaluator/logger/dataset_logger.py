from colorama import Fore
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.evaluator.logger.db_logger import log


def add_dataset_log(dataset_name, dataset_seed, dataset_size):
    sql_insert = " INSERT INTO dataset_log (dataset_id,batch_id,dataset_name,dataset_seed,dataset_size) " + \
          " VALUES (NULL,?,?,?,?)"
    sql_query = " SELECT dataset_id FROM dataset_log " \
                " WHERE batch_id = ? AND dataset_name = ? AND dataset_seed = ? AND dataset_size = ?"
    args = (str(batch_flag.batch_id), str(dataset_name), str(dataset_seed), str(dataset_size))
    result = log.query_log(sql_query, args)
    if result is not None:
        dataset_id = result["dataset_id"]
        if log.debug_log:
            msg = "[ LOGGER ] 数据集记录存在 本次测试Dataset_id为 {}".format(dataset_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return dataset_id
    else:
        dataset_id = log.insert_log(sql_insert, args)
        if log.debug_log:
            msg = "[ LOGGER ] 已写入日志 本次测试Dataset_id为 {}".format(dataset_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return dataset_id


def find_dataset_log(dataset_id):
    sql = " SELECT * FROM dataset_log WHERE dataset_id = ?"
    return log.query_log(sql,(dataset_id,))


def add_img_log(dataset_id, ori_img_label, ori_img_cursor):
    sql_insert = " INSERT INTO ori_img_log (ori_img_id, batch_id, dataset_id, ori_img_label, ori_img_cursor) " + \
                 " VALUES (NULL,?,?,?,?)"
    sql_query = " SELECT ori_img_id FROM ori_img_log " \
                " WHERE batch_id = ? AND dataset_id = ? AND ori_img_label = ? AND ori_img_cursor = ?"

    args = (str(batch_flag.batch_id), str(dataset_id), str(ori_img_label), str(ori_img_cursor))
    result = log.query_log(sql_query, args)
    if result is not None:
        img_id = result["ori_img_id"]
        if log.debug_log:
            msg = "[ LOGGER ] 日志存在 本次选定的图片img_id为 {}".format(img_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return img_id
    else:
        img_id = log.insert_log(sql_insert, args)
        if log.debug_log:
            msg = "[ LOGGER ] 已写入日志 本次选定的图片img_id为 {}".format(img_id)
            reporter.console_log(msg, Fore.CYAN, type="DEBUG")
        return img_id


def find_img_log(ori_img_id):
    sql = "SELECT * FROM ori_img_log WHERE ori_img_id = ?"
    return log.query_log(sql, (ori_img_id,))
