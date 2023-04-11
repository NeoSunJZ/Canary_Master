from colorama import Fore
from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.helper.realtime_reporter import reporter


def add_img_log(ori_img_label, ori_img_cursor):
    sql_insert = " INSERT INTO ori_img_log (ori_img_id, ori_img_label, ori_img_cursor) VALUES (NULL,?,?)"
    sql_query = " SELECT ori_img_id FROM ori_img_log WHERE ori_img_label = ? AND ori_img_cursor = ?"

    args = (str(ori_img_label), str(ori_img_cursor))
    result = task_manager.test_data_logger.query_log(sql_query, args)
    if result is not None:
        img_id = result["ori_img_id"]
    else:
        img_id = task_manager.test_data_logger.insert_log(sql_insert, args)
    if task_manager.test_data_logger.debug_log:
        msg = "[ LOGGER ] Logged. Now selected Img ID is {}".format(img_id)
        reporter.console_log(msg, Fore.CYAN, type="DEBUG")
    return img_id


def find_img_log_by_id(ori_img_id):
    sql = "SELECT * FROM ori_img_log WHERE ori_img_id = ?"
    return task_manager.test_data_logger.query_log(sql, (ori_img_id,))


def find_all_img_logs():
    sql = "SELECT * FROM ori_img_log "
    return task_manager.test_data_logger.query_logs(sql, ())

