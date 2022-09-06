import datetime
import sqlite3
import time

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.task_thread import task_thread


class SystemLog:

    def __init__(self):
        self.conn = sqlite3.connect('system_log.db', check_same_thread=False)
        self.conn.row_factory = self.dict_factory
        self.is_recovery_mode = False
        self.system_log_id = None
        self.step = None
        self.step_sequence = 0
        self.init()

    @staticmethod
    def dict_factory(cursor, row):
        data = {}
        for idx, col in enumerate(cursor.description):
            data[col[0]] = row[idx]
        return data

    def init(self):
        cursor = self.conn.cursor()
        cursor.execute('create table if not exists system_log '
                       '(id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'step varchar(40),'
                       'participant varchar(40),'
                       'completed_num integer,'
                       'is_finish varchar(40), '
                       'stop_reason varchar(40), '
                       'stop_time datetime)')

        cursor.execute('create table if not exists system_console_msg '
                       '(id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'msg varchar(40),'
                       'type varchar(40),'
                       'record_time datetime)')
        cursor.close()
        self.conn.commit()

    def start_recovery_mode(self, batch_id):
        self.is_recovery_mode = True
        batch_flag.set_batch(batch_id)

    def set_step(self, step, is_first=False):
        if is_first:
            self.step_sequence = 0
        self.step = step
        self.step_sequence += 1

    def new_record(self, participant):
        cursor = self.conn.cursor()
        sql_insert = " INSERT INTO system_log (id, batch_id, step, participant, completed_num, is_finish, stop_reason, stop_time) " + \
                     " VALUES (NULL,'{}','{}','{}',0,'False',NULL,NULL)" \
                         .format(str(batch_flag.batch_id), str(self.step), str(participant))
        cursor.execute(sql_insert)
        self.system_log_id = int(cursor.lastrowid)
        cursor.close()
        self.conn.commit()

    def update_completed_num(self, new_completed_num):
        cursor = self.conn.cursor()
        sql_update = "UPDATE system_log SET completed_num = completed_num + {} WHERE id = '{}'" \
            .format(new_completed_num, str(self.system_log_id))
        cursor.execute(sql_update)
        cursor.close()
        self.conn.commit()

        # 强行终止检查点
        task_thread.sys_check_task_status()

    def update_finish_status(self, is_finish=True, stop_reason=None):
        cursor = self.conn.cursor()
        if is_finish:
            stop_reason = "Task_End_Normally"
        sql_update = "UPDATE system_log SET is_finish = '{}', stop_reason = \"{}\", stop_time = '{}' WHERE id = {}" \
            .format(is_finish, stop_reason.replace('"', "'"), datetime.datetime.now(), str(self.system_log_id))
        print(sql_update)
        cursor.execute(sql_update)
        cursor.close()
        self.conn.commit()

    def check_skip(self, participant):
        if self.is_recovery_mode is not True:
            self.new_record(participant)
            return False, 0
        cursor = self.conn.cursor()
        sql_query = " SELECT * FROM system_log WHERE batch_id = '{}' AND step = '{}' AND participant = '{}'" \
            .format(str(batch_flag.batch_id), str(self.step), str(participant))
        cursor.execute(sql_query)
        value = cursor.fetchone()
        cursor.close()
        if value is None:
            self.new_record(participant)
            return False, 0
        elif value.get("is_finish", False):
            return True
        else:
            return False, value.get("completed_num")

    def get_all_task_progress_log(self, batch_id):
        cursor = self.conn.cursor()
        sql_query = " SELECT * FROM system_log WHERE batch_id = '{}'".format(str(batch_id))
        cursor.execute(sql_query)
        values = cursor.fetchall()
        cursor.close()
        return values

    def new_console_record(self, msg, type):
        cursor = self.conn.cursor()
        sql_insert = " INSERT INTO system_console_msg (id, batch_id, msg, type, record_time) " + \
                     " VALUES (NULL,'{}',\"{}\",'{}','{}')" \
                         .format(str(batch_flag.batch_id), str(msg.replace('"', "'")), str(type), datetime.datetime.now())
        cursor.execute(sql_insert)
        cursor.close()
        self.conn.commit()

    def get_all_console_msg(self, batch_id, before_time):
        cursor = self.conn.cursor()
        if before_time is None:
            before_time = datetime.datetime.now()
        else:
            before_time = time.strftime("%Y-%m-%d %H:%M:%S.%s", time.localtime(int(before_time) / 1000))
        sql_query = " SELECT * FROM system_console_msg WHERE batch_id = '{}' AND record_time < Datetime('{}') ".format(str(batch_id), str(before_time))
        cursor.execute(sql_query)
        values = cursor.fetchall()
        cursor.close()
        return values

global_system_log = SystemLog()