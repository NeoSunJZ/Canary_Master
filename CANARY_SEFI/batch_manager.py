import os
import random
import string

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.helper.system_log import SystemLog
from CANARY_SEFI.evaluator.logger.test_data_logger import TestDataLogger


class BatchManager(object):
    def __init__(self):
        self.batch_token = None
        self.test_data_logger = None
        self.sys_log_logger = None
        self.base_temp_path = None

    def init_batch(self, batch_token=None, show_logo=False):
        if batch_token == self.batch_token and batch_token is not None:
            return
        if show_logo:
            print_logo()
        if batch_token is None:
            self.batch_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        else:
            self.batch_token = batch_token

        self.base_temp_path = config_manager.config.get("baseTemp", "Raw_Data/") + self.batch_token + "/"
        if not os.path.exists(self.base_temp_path):
            os.makedirs(self.base_temp_path)

        self.test_data_logger = TestDataLogger(self.base_temp_path)
        self.sys_log_logger = SystemLog(self.base_temp_path)


batch_manager = BatchManager()
