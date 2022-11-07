import os
import random
import string

import torch

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.helper.system_log import SystemLog
from CANARY_SEFI.evaluator.logger.test_data_logger import TestDataLogger


class TaskManager(object):
    def __init__(self):
        self.task_token = None
        self.test_data_logger = None
        self.sys_log_logger = None
        self.base_temp_path = None
        self.run_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_task(self, task_token=None, show_logo=False, run_device=None):
        if task_token == self.task_token and task_token is not None:
            print("跳过初始化")
            return
        if show_logo:
            print_logo()
        if task_token is None:
            self.task_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        else:
            self.task_token = task_token

        self.base_temp_path = config_manager.config.get("baseTemp", "Raw_Data/") + self.task_token + "/"
        if not os.path.exists(self.base_temp_path):
            os.makedirs(self.base_temp_path)

        self.test_data_logger = TestDataLogger(self.base_temp_path)
        self.sys_log_logger = SystemLog(self.base_temp_path)

        self.run_device = self.run_device if run_device is None else run_device


task_manager = TaskManager()