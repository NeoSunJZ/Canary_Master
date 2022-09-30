import random
import string

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.evaluator.logger.test_data_logger import TestDataLogger


class BatchManager(object):
    def __init__(self):
        self.batch_token = None
        self.test_data_logger = None
        self.sys_log_logger = None

        self.init_batch()

    def init_batch(self):
        print_logo()
        self.batch_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        self.test_data_logger = TestDataLogger(self.batch_token)


batch_manager = BatchManager()
