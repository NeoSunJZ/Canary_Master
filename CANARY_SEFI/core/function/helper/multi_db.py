import os
import pickle
import sqlite3

from colorama import Fore
from tqdm import tqdm

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_all_img_logs
from CANARY_SEFI.evaluator.logger.indicator_data_handler import get_all_attack_deflection_capability_indicator_data, \
    save_attack_deflection_capability_indicator_data, get_all_attack_adv_example_da_indicator_data, \
    save_attack_adv_example_da_indicator_data, get_all_attack_adv_example_cost_indicator_data, \
    save_attack_adv_example_cost_indicator_data
from CANARY_SEFI.evaluator.logger.inference_test_data_handler import save_inference_test_data
from CANARY_SEFI.handler.tools.sqlite_db_logger import SqliteDBLogger
from CANARY_SEFI.task_manager import task_manager


def use_multi_database(center_database_token=None, mode=MultiDatabaseMode.SIMPLE, multi_database_config=None):
    task_manager.multi_database = MultiDatabase(center_database_token, mode, multi_database_config)


class MultiDatabase:
    def __init__(self, center_database_token=None, multi_database_mode=MultiDatabaseMode.SIMPLE,
                 multi_database_config=None):
        self.multi_database_mode = multi_database_mode
        self.center_database_token = center_database_token
        self.center_database = None

        if self.center_database_token is not None:
            self.is_currently_center = False
            self.connect_center_database()
        else:
            self.is_currently_center = True

        self.multi_database_config = multi_database_config if multi_database_config is not None else {}

    def connect_center_database(self):
        base_temp_path = config_manager.config.get("centerDatabase", config_manager.config.get("baseTemp", "Raw_Data/")) + self.center_database_token + "/"
        full_path = base_temp_path + "database/evaluator_logger.db"
        # 检查是否存在数据库文件
        if not os.path.exists(full_path):
            raise FileNotFoundError("[ Logic Error ] The center database file does not exist under the path!")
        center_database_conn = sqlite3.connect(full_path, check_same_thread=False)
        self.center_database = SqliteDBLogger(center_database_conn)

    def move_inference_test_data_from_center_database(self):
        if self.multi_database_config.get("not_move_inference_test_data_from_center_database", False):
            msg = "The 'move_inference_test_data_from_center_database' service is configured to disable"
            reporter.console_log(msg, Fore.RED, show_task=True, show_step_sequence=True)
            return
        if self.multi_database_mode is MultiDatabaseMode.EACH_ATTACK_ISOLATE_DB and not self.is_currently_center:
            # 检查是否满足迁移条件，对比图片数据是否一致（中心库的图片量不得少于分库，且存在的部分必须完全一致）
            center_ori_img_logs = self.center_database.query_logs("SELECT * FROM ori_img_log", ())
            now_ori_img_logs = find_all_img_logs()
            for now_ori_img_log in now_ori_img_logs:
                find_log = False
                for center_ori_img_log in center_ori_img_logs:
                    if now_ori_img_log['ori_img_id'] == center_ori_img_log['ori_img_id']:
                        if now_ori_img_log['ori_img_label'] == center_ori_img_log['ori_img_label'] and now_ori_img_log['ori_img_cursor'] == center_ori_img_log['ori_img_cursor']:
                            find_log = True
                            break
                        else:
                            raise RuntimeError(" [ Logic Error ] The original picture information stored in the central database and the current database is inconsistent!")
                if not find_log:
                    raise RuntimeError(" [ Logic Error ] The original picture information stored in the central database is less than that in the current database")
            # 检查完成，至此已经可以确保两个数据库的数据集一致性

            msg = "Sub-database data consistency check completed, preparing for migration.({} in total)"\
                .format(len(now_ori_img_logs))
            reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

            # 标记当前步骤
            task_manager.sys_log_logger.set_step(Step.MODEL_INFERENCE_CAPABILITY_TEST_MIGRATED)
            is_skip, completed_num = global_recovery.check_skip("DATABASE_" + self.center_database_token)
            if is_skip:
                return None
            # inference_test_data 迁移
            center_inference_test_data = self.center_database.query_logs("SELECT * FROM inference_test_data", ())
            move_logs = {}
            with tqdm(total=len(center_inference_test_data), desc="Progress", ncols=120) as bar:
                for log in center_inference_test_data:
                    save_inference_test_data(img_id=log["img_id"],
                                            img_type=log["img_type"],
                                            inference_model=log["inference_model"],
                                            inference_img_label=log["inference_img_label"],
                                            inference_img_conf_array=log["inference_img_conf_array"],
                                            inference_cams=(log["true_class_cams"], log["inference_class_cams"]),
                                            use_pickle_dump=False)
                    if move_logs.get(log["inference_model"], None) is None:
                        move_logs[log["inference_model"]] = 1
                    else:
                        move_logs[log["inference_model"]] += 1
                    bar.update(1)

            # 记录日志
            for model_name in move_logs.keys():
                msg = "Model({})'s Inference test record migration completed, amount {}".format(model_name, move_logs[model_name])
                reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

            task_manager.sys_log_logger.update_completed_num(len(center_inference_test_data))
            task_manager.sys_log_logger.update_finish_status(True)
        else:
            raise RuntimeError(" [ Logic Error ] Illegal call!")

    def move_indicator_data_to_center_database(self):
        # attack_deflection_capability_indicator_data 迁移
        attack_deflection_capability_indicator_data = get_all_attack_deflection_capability_indicator_data()
        counts = 0
        with tqdm(total=len(attack_deflection_capability_indicator_data), desc="Progress", ncols=120) as bar:
            for log in attack_deflection_capability_indicator_data:
                save_attack_deflection_capability_indicator_data(atk_name=log["atk_name"],
                                                                 base_model=log["base_model"],
                                                                 inference_model=log["inference_model"],
                                                                 adv_example_file_type=log["adv_example_file_type"],
                                                                 MR=log["MR"],
                                                                 AIAC=log["AIAC"],
                                                                 ARTC=log["ARTC"],
                                                                 ACAMC_A=log["ACAMC_A"],
                                                                 ACAMC_T=log["ACAMC_T"],
                                                                 atk_perturbation_budget=log["atk_perturbation_budget"],
                                                                 logger=self.center_database)
                counts += 1
                bar.update(1)

        msg = "Attack Deflection Capability Indicator Data migration completed, amount {}".format(counts)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

        # attack_adv_example_da_indicator_data 迁移
        attack_adv_example_da_indicator_data = get_all_attack_adv_example_da_indicator_data()
        counts = 0
        with tqdm(total=len(attack_adv_example_da_indicator_data), desc="Progress", ncols=120) as bar:
            for log in attack_adv_example_da_indicator_data:
                save_attack_adv_example_da_indicator_data(atk_name=log["atk_name"],
                                                          base_model=log["base_model"],
                                                          adv_example_file_type=log["adv_example_file_type"],
                                                          AMD=log["AMD"],
                                                          AED=log["AED"],
                                                          AED_HF=log["AED_HF"],
                                                          AED_LF=log["AED_LF"],
                                                          APCR=log["APCR"],
                                                          ADMS=log["ADMS"],
                                                          ALMS=log["ALMS"],
                                                          atk_perturbation_budget=log["atk_perturbation_budget"],
                                                          logger=self.center_database)
                counts += 1
                bar.update(1)

        msg = "Attack Adv Example DA Indicator Data migration completed, amount {}".format(counts)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

        # attack_deflection_capability_indicator_data 迁移
        attack_adv_example_cost_indicator_data = get_all_attack_adv_example_cost_indicator_data()
        counts = 0
        with tqdm(total=len(attack_adv_example_cost_indicator_data), desc="Progress", ncols=120) as bar:
            for log in attack_adv_example_cost_indicator_data:
                save_attack_adv_example_cost_indicator_data(atk_name=log["atk_name"],
                                                            base_model=log["base_model"],
                                                            ACT=log["ACT"],
                                                            AQN_F=log["AQN_F"],
                                                            AQN_B=log["AQN_B"],
                                                            atk_perturbation_budget=log["atk_perturbation_budget"],
                                                            logger=self.center_database)
                counts += 1
                bar.update(1)

        msg = "Attack Adv Example Cost Indicator Data migration completed, amount {}".format(counts)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    def get_skip_step(self):
        skip_step_list = {
            "model_inference_capability_test_and_evaluation": False,
            "adv_example_generate": False,
            "attack_test_and_evaluation": False,
            "synthetical_capability_evaluation": False,
        }
        if self.multi_database_mode is MultiDatabaseMode.SIMPLE or self.multi_database_mode is None:
            # SIMPLE模式，默认全部执行
            pass
        elif self.multi_database_mode is MultiDatabaseMode.EACH_ATTACK_ISOLATE_DB:
            # EACH_ATTACK_ISOLATE_DB模式，中心库任务执行模型推理能力测试与评估，非中心库不执行
            # EACH_ATTACK_ISOLATE_DB模式，中心库任务不执行生成对抗样本的过程，非中心库执行
            # EACH_ATTACK_ISOLATE_DB模式，中心库任务不执行对抗样本的测试与评估，非中心库执行
            # EACH_ATTACK_ISOLATE_DB模式，中心库任务执行综合能力评估，非中心库不执行
            if self.is_currently_center:
                # 中心库
                skip_step_list["adv_example_generate"] = True
                skip_step_list["attack_test_and_evaluation"] = True

                if self.multi_database_config.get("not_init", False):
                    # 非首次初始化，直接不执行模型推理能力测试与评估
                    skip_step_list["model_inference_capability_test_and_evaluation"] = True
                else:
                    # 首次初始化，直接不执行综合能力评估
                    skip_step_list["synthetical_capability_evaluation"] = True
            else:
                # 子库
                skip_step_list["model_inference_capability_test_and_evaluation"] = True
                skip_step_list["synthetical_capability_evaluation"] = True
                # 迁移中心数据库测评信息到子库以完成评估
                self.move_inference_test_data_from_center_database()
        else:
            raise RuntimeError(" [ Logic Error ] Unexpected mode!")

        msg = "The current multi-database mode: {}, \n" \
              "The current database: {}, \n" \
              "The steps to be skipped:{}" \
            .format(self.multi_database_mode, "Center Database" if self.is_currently_center else "Sub Database",
                    list(key for key, value in skip_step_list.items() if value is True))
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

        def callback():
            if self.multi_database_mode is MultiDatabaseMode.SIMPLE or self.multi_database_mode is None:
                # SIMPLE模式，无回调
                pass
            elif self.multi_database_mode is MultiDatabaseMode.EACH_ATTACK_ISOLATE_DB:
                if self.is_currently_center:
                    pass  # EACH_ATTACK_ISOLATE_DB模式中心库，无回调
                else:
                    # EACH_ATTACK_ISOLATE_DB模式子库，迁移结果数据回中心库
                    # 迁移子数据库测评信息到中心库以完成评估
                    self.move_indicator_data_to_center_database()

        return skip_step_list, callback
