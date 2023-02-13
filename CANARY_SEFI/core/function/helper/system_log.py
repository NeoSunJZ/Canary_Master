import datetime
import os
import sqlite3
import time
from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.handler.tools.sqlite_db_logger import SqliteDBLogger


class SystemLog(SqliteDBLogger):

    def __init__(self, base_temp_path):
        # 检查是否存在数据库文件
        if not os.path.exists(base_temp_path + "database/"):
            os.makedirs(base_temp_path + "database/")
        full_path = base_temp_path + "database/system_logger.db"

        conn = sqlite3.connect(full_path, check_same_thread=False)
        SqliteDBLogger.__init__(self, conn)

        if not os.path.exists(full_path):
            self.init_database()

        self.system_log_id = None
        self.step = None
        self.step_sequence = 0

    def init_database(self):
        cursor = self.conn.cursor()
        cursor.execute('create table if not exists system_log '
                       '(id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'step varchar(40),'
                       'participant varchar(40),'
                       'completed_num integer,'
                       'is_finish varchar(40), '
                       'stop_reason varchar(40), '
                       'stop_time datetime)')

        cursor.execute('create table if not exists system_console_msg '
                       '(id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'msg varchar(40),'
                       'type varchar(40),'
                       'record_time datetime)')
        cursor.close()
        self.conn.commit()

    def set_step(self, step, is_first=False):
        if is_first:
            self.step_sequence = 0
        self.step = step.name
        self.step_sequence += 1

    def new_record(self, participant):
        cursor = self.conn.cursor()
        sql_insert = " INSERT INTO system_log (id, step, participant, completed_num, is_finish, stop_reason, stop_time) " + \
                     " VALUES (NULL, ?, ?, 0, False, NULL, NULL)"
        cursor.execute(sql_insert, (str(self.step), str(participant)))
        self.system_log_id = int(cursor.lastrowid)
        cursor.close()
        self.conn.commit()

    def update_completed_num(self, new_completed_num):
        cursor = self.conn.cursor()
        sql_update = "UPDATE system_log SET completed_num = completed_num + ? WHERE id = ? "
        cursor.execute(sql_update, (new_completed_num, str(self.system_log_id)))
        cursor.close()
        self.conn.commit()

        # 强行终止检查点
        task_thread.sys_check_task_status()

    def update_finish_status(self, is_finish=True, stop_reason=None, is_restart=False):
        cursor = self.conn.cursor()
        if is_restart:
            sql_update_restart = "UPDATE system_log SET is_finish = '0', stop_reason = NULL, stop_time = NULL WHERE id = ? "
            cursor.execute(sql_update_restart, (str(self.system_log_id), ))
        else:
            if is_finish:
                stop_reason = "Task_End_Normally"
            sql_update_normal = "UPDATE system_log SET is_finish = ?, stop_reason = ?, stop_time = ? WHERE id = ? "
            cursor.execute(sql_update_normal, (is_finish , str(stop_reason), datetime.datetime.now(), str(self.system_log_id)))
        cursor.close()
        self.conn.commit()

    def get_current_step_progress_log(self, participant):
        cursor = self.conn.cursor()
        sql_query = " SELECT * FROM system_log WHERE step = ? AND participant = ?"
        cursor.execute(sql_query, (str(self.step), str(participant)))
        value = cursor.fetchone()
        cursor.close()
        return value

    def get_all_task_progress_log(self):
        cursor = self.conn.cursor()
        sql_query = " SELECT * FROM system_log "
        cursor.execute(sql_query, ())
        values = cursor.fetchall()
        for value in values:
            value["step_name"] = Step[value["step"]].step_name()
        cursor.close()
        return values

    def new_console_record(self, msg, type):
        cursor = self.conn.cursor()
        sql_insert = " INSERT INTO system_console_msg (id, msg, type, record_time) VALUES (NULL, ?, ?, ?)"
        cursor.execute(sql_insert, (str(msg), str(type), datetime.datetime.now()))
        cursor.close()
        self.conn.commit()

    def get_all_console_msg(self, before_time):
        cursor = self.conn.cursor()
        if before_time is None:
            before_time = datetime.datetime.now()
        else:
            before_time = time.strftime("%Y-%m-%d %H:%M:%S.%s", time.localtime(int(before_time) / 1000))
        sql_query = " SELECT * FROM system_console_msg WHERE record_time < Datetime(?) "
        cursor.execute(sql_query, (str(before_time),))
        values = cursor.fetchall()
        cursor.close()
        return values
