import os
import random
import shutil
import string

import torch
from colorama import Fore

from CANARY_SEFI.compatibility_repair.repair import compatibility_repair
from CANARY_SEFI.copyright import print_logo, get_system_version
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.helper.system_log import SystemLog
from CANARY_SEFI.evaluator.logger.test_data_logger import TestDataLogger
from CANARY_SEFI.handler.json_handler.json_io_handler import get_info_from_json_file, save_info_to_json_file


class TaskManager(object):
    def __init__(self):
        self.task_token = None
        self.test_data_logger = None
        self.sys_log_logger = None
        self.base_temp_path = None
        self.run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_status = False
        # 分批次核心数据库构建模式
        self.multi_database = None

    def init_task(self, task_token=None, show_logo=False, run_device=None, not_retain_same_token=False, logo_color=Fore.GREEN):
        if self.init_status is True:
            raise RuntimeError("[ Logic Error ] Duplicate initialization!")
        if task_token == self.task_token and task_token is not None:
            print("Initialization skipped!")
            return
        if show_logo:
            print_logo(color=logo_color)
        if task_token is None:
            self.task_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        else:
            self.task_token = task_token

        self.base_temp_path = config_manager.config.get("baseTemp", "Raw_Data/") + self.task_token + "/"
        if not os.path.exists(self.base_temp_path):
            os.makedirs(self.base_temp_path)
        elif not_retain_same_token:
            # 不保留生效时需要先删除再新建
            shutil.rmtree(self.base_temp_path)
            os.makedirs(self.base_temp_path)

        self.test_data_logger = TestDataLogger(self.base_temp_path)
        self.sys_log_logger = SystemLog(self.base_temp_path)

        system_info = get_info_from_json_file(task_manager.base_temp_path, "system_info.json")
        config = get_info_from_json_file(task_manager.base_temp_path, "config.json")

        if system_info is None and config is None:
            self.set_system_info()
        else:
            if system_info is None:
                system_info = {}
            version = system_info.get("version", None)
            compatibility_repair(version, {
                "test_data_logger": self.test_data_logger,
                "sys_log_logger": self.sys_log_logger
            })
            self.set_system_info()

        self.run_device = self.run_device if run_device is None else run_device

        self.init_status = True

    def set_system_info(self):
        system_info = {
            "version": get_system_version()
        }
        save_info_to_json_file(system_info, self.base_temp_path, "system_info.json")


task_manager = TaskManager()
