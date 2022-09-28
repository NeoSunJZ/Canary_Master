import sqlite3

class Logger:
    def __init__(self):
        self.conn = sqlite3.connect('Benchmark/Benchmark_main.db', check_same_thread=False)
        self.conn.row_factory = self.dict_factory
        self.init()

    @staticmethod
    def dict_factory(cursor, row):
        data = {}
        for idx, col in enumerate(cursor.description):
            data[col[0]] = row[idx]
        return data

    def init(self):
        cursor = self.conn.cursor()

        cursor.execute('create table if not exists benchmark_info_log '
                       '(benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'dataset_name varchar(20), '
                       'dataset_seed varchar(20))')

        cursor.execute('create table if not exists benchmark_adv_total_log '
                       '(benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'dateset_size integer(8), '
                       'atk_name varchar(20), '
                       'base_model varchar(20), '
                       'test_model varchar(20))')

        cursor.execute('create table if not exists benchmark_adv_batch_log '
                       '(benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'batch_id varchar(20), '
                       'dateset_start_cursor integer(8), '
                       'dateset_size integer(8), '
                       'atk_name varchar(20), '
                       'base_model varchar(20), '
                       'test_model varchar(20))')

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
                       'atk_perturbation_budget float,'
                       'test_model_name varchar(8), '
                       'misclassification_ratio float, '
                       'average_increase_adversarial_class_confidence float, '
                       'average_reduction_true_class_confidence float, '
                       'average_cost_time float)')

        cursor.execute('create table if not exists test_report_adv_da_capability '
                       '(batch_id integer(8), '
                       'atk_name varchar(8), '
                       'base_model varchar(20), '
                       'atk_perturbation_budget float,'
                       'average_maximum_disturbance varchar, '
                       'average_euclidean_distortion varchar, '
                       'average_pixel_change_ratio varchar, '
                       'average_deep_metrics_similarity varchar, '
                       'average_low_level_metrics_similarity varchar)')

        cursor.execute('create table if not exists model_security_synthetical_capability '
                       '(batch_id integer(8), '
                       'model_name varchar(8), '
                       'clear_acc float, '
                       'clear_f1 float, '
                       'clear_conf integer, '
                       'misclassification_ratio varchar(20), '
                       'average_increase_adversarial_class_confidence varchar, '
                       'average_reduction_true_class_confidence varchar, '
                       'average_cost_time float,'
                       'average_maximum_disturbance varchar, '
                       'average_euclidean_distortion varchar, '
                       'average_pixel_change_ratio varchar, '
                       'average_deep_metrics_similarity varchar, '
                       'average_low_level_metrics_similarity varchar)')

        cursor.execute('create table if not exists attack_synthetical_capability '
                       '(batch_id integer(8), '
                       'attack_name varchar(8), '
                       'average_clear_acc float, '
                       'average_clear_f1 float, '
                       'average_clear_conf integer, '
                       'misclassification_ratio varchar(20), '
                       'average_increase_adversarial_class_confidence varchar, '
                       'average_reduction_true_class_confidence varchar, '
                       'average_cost_time float,'
                       'average_maximum_disturbance varchar, '
                       'average_euclidean_distortion varchar, '
                       'average_pixel_change_ratio varchar, '
                       'average_deep_metrics_similarity varchar, '
                       'average_low_level_metrics_similarity varchar)')


        cursor.close()
        self.conn.commit()

    def insert_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        log_id = int(cursor.lastrowid)
        cursor.close()
        self.conn.commit()
        return log_id

    def query_logs(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        values = cursor.fetchall()
        cursor.close()
        return values

    def query_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        values = cursor.fetchone()
        cursor.close()
        return values

    def update_log(self, sql, args):
        cursor = self.conn.cursor()
        cursor.execute(sql, args)
        cursor.close()
        self.conn.commit()

    def finish(self):
        self.conn.commit()

log = Logger()