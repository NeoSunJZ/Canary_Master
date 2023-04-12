import os
import sqlite3

from canary_sefi.task_manager import task_manager


def query_logs(conn, sql, args):
    cursor = conn.cursor()
    cursor.execute(sql, args)
    values = cursor.fetchall()
    cursor.close()
    return values


if __name__ == "__main__":
    task_manager.init_task(show_logo=True, run_device="cuda")
    base_temp_path = input("Please enter the result file path of the OLD Version: ")
    full_path = base_temp_path + "database/evaluator_logger.db"
    # 检查是否存在数据库文件
    if not os.path.exists(full_path):
        raise FileNotFoundError("The database file does not exist under the path!")

    conn_old = sqlite3.connect(full_path, check_same_thread=False)

    # adv_img_file_log 迁移
    adv_img_file_log_old = query_logs(conn_old, "SELECT * FROM adv_img_file_log", ())
    for log in adv_img_file_log_old:
        task_manager.test_data_logger.\
            insert_log("INSERT INTO adv_img_file_log (adv_img_file_id, attack_id, cost_time, ori_img_id, adv_img_filename, "
                       "adv_raw_nparray_filename) VALUES (?,?,?,?,?,?)", log)

    # attack_info_log 迁移
    attack_info_log_old = query_logs(conn_old, "SELECT * FROM attack_info_log", ())
    for log in attack_info_log_old:
        task_manager.test_data_logger.\
            insert_log("INSERT INTO attack_info_log (attack_id, atk_name, base_model, atk_type, atk_perturbation_budget) " + \
                 " VALUES (?,?,?,?,?)", log)

    # ori_img_log 迁移
    ori_img_log_old = query_logs(conn_old, "SELECT * FROM ori_img_log", ())
    for log in ori_img_log_old:
        task_manager.test_data_logger.\
            insert_log("INSERT INTO ori_img_log (ori_img_id, ori_img_label, ori_img_cursor) VALUES (?,?,?)", log)

    print("Successfully migrated!")
