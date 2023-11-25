import os
import random
import shutil
import string

import torch
from colorama import Fore

from canary_sefi.compatibility_repair.repair import compatibility_repair
from canary_sefi.copyright import print_logo, get_system_version
from canary_sefi.core.config.config_manager import config_manager
from canary_sefi.core.function.helper.system_log import SystemLog
from canary_sefi.evaluator.logger.test_data_logger import TestDataLogger
from canary_sefi.handler.json_handler.json_io_handler import get_info_from_json_file, save_info_to_json_file


class TaskManager(object):
    def __init__(self):
        self.task_token = None
        self.monitor_token = None
        self.test_data_logger = None
        self.sys_log_logger = None
        self.base_temp_path = None
        self.run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 分批次核心数据库构建模式
        self.multi_database = None

    def init_task(self, task_token=None, show_logo=False, run_device=None, not_retain_same_token=False,
                  logo_color=Fore.GREEN, is_fast_test=False):
        if task_token == self.task_token and task_token is not None:
            print("Initialization skipped!")
            return
        if show_logo:
            print_logo(color=logo_color)
        if task_token is None:
            task_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))

        if is_fast_test:
            task_token = 'FAST_TASK_' + task_token

        base_temp_path = self.get_base_temp_path(task_token)
        if not os.path.exists(base_temp_path) or not_retain_same_token is True:
            if not_retain_same_token:
                # 不保留生效时需要先删除
                shutil.rmtree(base_temp_path)
            # 新建文件夹
            os.makedirs(base_temp_path)
            self.set_system_info(base_temp_path)
            self.load_task(task_token, run_device)
        else:
            self.load_task(task_token, run_device)

        return task_token

    def load_task(self, task_token, run_device=None):
        if task_token == self.task_token:
            print("Initialization skipped!")
            return

        base_temp_path = self.get_base_temp_path(task_token)
        if os.path.exists(base_temp_path):
            self.task_token = task_token
            self.base_temp_path = base_temp_path
            self.test_data_logger = TestDataLogger(self.base_temp_path)
            self.sys_log_logger = SystemLog(self.base_temp_path)
            self.run_device = self.run_device if run_device is None else run_device
            self.version_control_check()
        else:
            raise FileNotFoundError("[SEFI] Task core data folder not find!")

    def version_control_check(self):
        system_info = get_info_from_json_file(self.base_temp_path, "system_info.json")
        version = None if system_info is None else system_info.get("version", None)
        compatibility_repair(version, {
            "test_data_logger": self.test_data_logger,
            "sys_log_logger": self.sys_log_logger
        })
        self.set_system_info(self.base_temp_path)

    @staticmethod
    def get_base_temp_path(task_token):
        return config_manager.config.get("baseTempPath", "Raw_Data/") + task_token + "/"

    @staticmethod
    def set_system_info(base_temp_path):
        system_info = {
            "version": get_system_version()
        }
        save_info_to_json_file(system_info, base_temp_path, "system_info.json")


task_manager = TaskManager()
