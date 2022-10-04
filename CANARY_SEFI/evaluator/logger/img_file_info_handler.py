from colorama import Fore
from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

logger = batch_manager.test_data_logger


def add_img_log(ori_img_label, ori_img_cursor):
    sql_insert = " INSERT INTO ori_img_log (ori_img_id, ori_img_label, ori_img_cursor) VALUES (NULL,?,?,?)"
    sql_query = " SELECT ori_img_id FROM ori_img_log WHERE ori_img_label = ? AND ori_img_cursor = ?"

    args = (str(ori_img_label), str(ori_img_cursor))
    result = logger.query_log(sql_query, args)
    if result is not None:
        img_id = result["ori_img_id"]
    else:
        img_id = logger.insert_log(sql_insert, args)
    if logger.debug_log:
        msg = "[ LOGGER ] Logged. Now selected Img ID is {}".format(img_id)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return img_id


def find_img_log_by_id(ori_img_id):
    sql = "SELECT * FROM ori_img_log WHERE ori_img_id = ?"
    return logger.query_log(sql, (ori_img_id,))


# def add_dataset_log(dataset_name, dataset_seed, dataset_size):
#     sql_insert = " INSERT INTO dataset_log (dataset_id,batch_id,dataset_name,dataset_seed,dataset_size) " + \
#           " VALUES (NULL,?,?,?,?)"
#     sql_query = " SELECT dataset_id FROM dataset_log " \
#                 " WHERE batch_id = ? AND dataset_name = ? AND dataset_seed = ? AND dataset_size = ?"
#     args = (str(batch_flag.batch_id), str(dataset_name), str(dataset_seed), str(dataset_size))
#     result = log.query_log(sql_query, args)
#     if result is not None:
#         dataset_id = result["dataset_id"]
#         if log.debug_log:
#             msg = "[ LOGGER ] 数据集记录存在 本次测试Dataset_id为 {}".format(dataset_id)
#             reporter.console_log(msg, Fore.CYAN, type="DEBUG")
#         return dataset_id
#     else:
#         dataset_id = log.insert_log(sql_insert, args)
#         if log.debug_log:
#             msg = "[ LOGGER ] 已写入日志 本次测试Dataset_id为 {}".format(dataset_id)
#             reporter.console_log(msg, Fore.CYAN, type="DEBUG")
#         return dataset_id
#
#
# def find_dataset_log(dataset_id):
#     sql = " SELECT * FROM dataset_log WHERE dataset_id = ?"
#     return log.query_log(sql,(dataset_id,))
