import sqlite3

class Logger:
    def __init__(self):
        self.conn = sqlite3.connect('logger.db')
        self.debug_log = True
        self.init()

    def init(self):
        cursor = self.conn.cursor()
        cursor.execute('create table if not exists adv_example_log '
                       '(adv_img_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'attack_id integer(8), '
                       'cost_time float, '
                       
                       'ori_img_id integer(8), '
                       'adv_img_filename varchar(40), '
                       
                       'adv_img_maximum_disturbance varchar, '
                       'adv_img_euclidean_distortion varchar, '
                       'adv_img_pixel_change_ratio varchar, '
                       'adv_img_deep_metrics_similarity varchar, '
                       'adv_img_low_level_metrics_similarity varchar)')

        cursor.execute('create table if not exists dataset_log '
                       '(dataset_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'dataset_name varchar(20), '
                       'dataset_seed varchar(20), '
                       'dataset_size varchar(20))')

        cursor.execute('create table if not exists attack_log '
                       '(attack_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'atk_name  varchar(20), '
                       'base_model varchar(20), '
                       'atk_type varchar(20),'
                       'atk_perturbation_budget float)')

        cursor.execute('create table if not exists ori_img_log '
                       '(ori_img_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'dataset_id integer(8), '
                       'ori_img_label integer(8), '
                       'ori_img_cursor varchar(20))')

        cursor.execute('create table if not exists inference_result '
                       '(inference_result_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'img_id integer, '
                       'img_type varchar(20), '
                       'batch_id varchar(20), '
                       'inference_model integer(8), '
                       'inference_img_label integer(8), '
                       'inference_img_conf_array varchar)')

        cursor.execute('create table if not exists test_report_model_capability '
                       '(batch_id integer(8), '
                       'model_name varchar(8), '
                       'clear_acc float, '
                       'clear_f1 float, '
                       'clear_conf integer)')

        cursor.execute('create table if not exists test_report_attack_capability '
                       '(batch_id integer(8), '
                       'atk_name varchar(8), '
                       'base_model varchar(20), '
                       'test_model_name varchar(8), '
                       'misclassification_ratio float, '
                       'average_increase_adversarial_class_confidence float, '
                       'average_reduction_true_class_confidence float, '
                       'average_cost_time float)')

        cursor.execute('create table if not exists test_report_adv_da_capability '
                       '(batch_id integer(8), '
                       'atk_name varchar(8), '
                       'base_model varchar(20), '
                       'average_maximum_disturbance varchar, '
                       'average_euclidean_distortion varchar, '
                       'average_pixel_change_ratio varchar, '
                       'average_deep_metrics_similarity varchar, '
                       'average_low_level_metrics_similarity varchar)')

        cursor.close()
        self.conn.commit()

    def insert_log(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        log_id = int(cursor.lastrowid)
        cursor.close()
        self.conn.commit()
        return log_id

    def query_log(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        values = cursor.fetchall()
        cursor.close()
        return values

    def update_log(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        cursor.close()
        self.conn.commit()

    def finish(self):
        self.conn.commit()
        self.conn.close()

log = Logger()