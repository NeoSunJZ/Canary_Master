import os
import sqlite3

from canary_sefi.handler.tools.sqlite_db_logger import SqliteDBLogger


class TestDataLogger(SqliteDBLogger):
    def __init__(self, base_temp_path):
        # 检查是否存在数据库文件
        if not os.path.exists(base_temp_path + "database/"):
            os.makedirs(base_temp_path + "database/")
        full_path = base_temp_path + "database/evaluator_logger.db"
        exist_db = os.path.exists(full_path)

        conn = sqlite3.connect(full_path, check_same_thread=False)
        SqliteDBLogger.__init__(self, conn)

        if not exist_db:
            self.init_database()

    def init_database(self):
        cursor = self.conn.cursor()

        cursor.execute('create table if not exists ori_img_log '
                       '(ori_img_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'ori_img_label integer, '
                       'ori_img_cursor varchar,'
                       'UNIQUE (ori_img_cursor))')

        cursor.execute('create table if not exists attack_info_log '
                       '(attack_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'atk_name  varchar, '
                       'base_model varchar, '
                       'atk_type varchar, '
                       'atk_perturbation_budget float,'
                       'UNIQUE (atk_name, base_model, atk_type, atk_perturbation_budget))')

        cursor.execute('create table if not exists adv_img_file_log '
                       '(adv_img_file_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'attack_id integer, '
                       'cost_time float, '
                       'ori_img_id integer, '
                       'query_num_forward integer, '
                       'query_num_backward integer, '
                       'adv_img_filename varchar, '
                       'adv_raw_nparray_filename varchar,'
                       'ground_valid varchar,'
                       'tlabel varchar, '
                       'UNIQUE (attack_id, ori_img_id))')

        cursor.execute('create table if not exists adv_trans_img_file_log '
                       '(adv_trans_img_file_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'trans_name varchar, '
                       'attack_id integer, '
                       'adv_img_file_id integer, '
                       'adv_trans_img_filename varchar, '
                       'adv_trans_raw_nparray_filename varchar, '
                       'ground_valid varchar,'
                       'UNIQUE (trans_name, adv_img_file_id))')

        cursor.execute('create table if not exists adv_example_da_test_data '
                       '(adv_img_file_id integer, '
                       'adv_example_file_type varchar, '
                       'maximum_disturbance float, '
                       'euclidean_distortion float, '
                       'high_freq_euclidean_distortion float, '
                       'low_freq_euclidean_distortion float, '
                       'pixel_change_ratio float, '
                       'deep_metrics_similarity float, '
                       'low_level_metrics_similarity float, '
                       'UNIQUE (adv_img_file_id,adv_example_file_type))')  # 每个对抗样本（不同转储文件类型）最多有一条记录，会覆盖

        cursor.execute('create table if not exists inference_test_data '
                       '(inference_test_data_id INTEGER PRIMARY KEY AUTOINCREMENT, '
                       'img_id integer, '
                       'img_type varchar, '
                       'inference_model varchar, '
                       'inference_img_label integer, '
                       'inference_img_conf_array varchar, '
                       'true_class_cams varchar, '
                       'inference_class_cams varchar)')

        cursor.execute('create table if not exists model_inference_capability_indicator_data '
                       '(model_name varchar PRIMARY KEY, '
                       'clear_acc float, '
                       'clear_f1 float, '
                       'clear_conf integer)')

        # 缩写释义:
        # MR: misclassification ratio
        # AIAC: average increase adversarial-class confidence
        # ARTC: average reduction true-class confidence
        # ACAMC_A: average G-CAM Change (Adversarial-class)
        # ACAMC_T: average G-CAM Change (True-class)
        cursor.execute('create table if not exists attack_deflection_capability_indicator_data '
                       '(atk_name varchar, '
                       'base_model varchar, '
                       'atk_perturbation_budget float, '
                       'inference_model varchar, '
                       'adv_example_file_type varchar, '  # 对抗样本文件类型 IMG文件可能导致真实误差，NP文件不会转换类型因而没有误差，但并不真实
                       'MR float, '
                       'AIAC float, '
                       'ARTC float, '
                       'ACAMC_A float, '
                       'ACAMC_T float, '
                       'UNIQUE (atk_name, base_model, atk_perturbation_budget, inference_model, adv_example_file_type))')

        cursor.execute('create table if not exists trans_deflection_capability_indicator_data '
                       '(atk_name varchar, '
                       'base_model varchar, '
                       'atk_perturbation_budget float, '
                       'trans_name varchar, '
                       'inference_model varchar, '
                       'trans_file_type varchar, '  # 防御样本文件类型 IMG文件可能导致真实误差，NP文件不会转换类型因而没有误差，但并不真实
                       'MR float, '
                       'AIAC float, '
                       'ARTC float, '
                       'ACAMC_A float, '
                       'ACAMC_T float, '
                       'UNIQUE (atk_name, base_model, atk_perturbation_budget, trans_name, inference_model, trans_file_type))')

        # 缩写释义:
        # ACT: average cost time
        # AQN_F: average query number (Forward)
        # AQN_B: average query number (Backward)
        cursor.execute('create table if not exists attack_adv_example_cost_indicator_data '
                       '(atk_name varchar, '
                       'base_model varchar, '
                       'atk_perturbation_budget float, '
                       'ACT float, '
                       'AQN_F integer, '
                       'AQN_B integer, '
                       'UNIQUE (atk_name, base_model, atk_perturbation_budget))')


        # 缩写释义:
        # AMD: average maximum disturbance
        # AED: average euclidean distortion
        # APCR: average pixel change ratio
        # ADMS: average deep metrics similarity
        # ALMS: average low-level metrics similarity
        cursor.execute('create table if not exists attack_adv_example_da_indicator_data '
                       '(atk_name varchar, '
                       'base_model varchar, '
                       'atk_perturbation_budget float, '
                       'adv_example_file_type varchar, '  # 对抗样本文件类型 IMG文件可能导致真实误差，NP文件不会转换类型因而没有误差，但并不真实
                       'AMD float, '
                       'AED float, '
                       'AED_HF float, '
                       'AED_LF float, '
                       'APCR float, '
                       'ADMS float, '
                       'ALMS float, '
                       'UNIQUE (atk_name, base_model, atk_perturbation_budget, adv_example_file_type))')


        cursor.execute('create table if not exists model_dimension_summary '
                       '(model_name varchar, '
                       'test_adv_example_file_type varchar, '
                       'ACC float, '
                       'F1 float, '
                       'Conf float, '
                       'MR float, '
                       'TAS float, '
                       'AIAC float, '
                       'ARTC float, '
                       'ACAMC_A float, '
                       'ACAMC_T float, '
                       'ACT float,'
                       'AQN_F integer, '
                       'AQN_B integer, '
                       'AMD float, '
                       'AED float, '
                       'AED_HF float, '
                       'AED_LF float, '
                       'APCR float, '
                       'ADMS float, '
                       'ALMS float)')

        cursor.execute('create table if not exists attack_dimension_summary '
                       '(attack_name varchar(8), '
                       'test_adv_example_file_type varchar, '
                       'ACC float, '
                       'F1 float, '
                       'Conf float, '
                       'MR float, '
                       'TAS float, '
                       'AIAC float, '
                       'ARTC float, '
                       'ACAMC_A float, '
                       'ACAMC_T float, '
                       'OTR_MR float, '
                       'OTR_AIAC float, ' 
                       'OTR_ARTC float, '
                       'ACT float,'
                       'AQN_F integer, '
                       'AQN_B integer, '
                       'AMD float, '
                       'AED float, '
                       'AED_HF float, '
                       'AED_LF float, '
                       'APCR float, '
                       'ADMS float, '
                       'ALMS float)')

        cursor.execute('create table if not exists defense_model_normal_inference_capability_indicator_data '
                       '(model_name varchar PRIMARY KEY, '
                       'CAV float, '
                       'RRSR float, '
                       'ConV float,'
                       'COS float)')

        cursor.execute('create table if not exists defense_model_adv_inference_capability_indicator_data '
                       '(model_name varchar PRIMARY KEY, '
                       'attack_name varchar, '
                       'DCAV float, '
                       'TCAV float)')


        cursor.close()
        self.conn.commit()
