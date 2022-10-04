import random

from colorama import Fore

from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo


def init_dataset(dataset_name, dataset_size, dataset_seed=None):
    # 标记当前步骤
    batch_manager.sys_log_logger.set_step(Step.INIT, is_first=True)

    msg = "SEFI Initialization"
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    dataset_seed = dataset_seed_handler(dataset_seed)
    # 构建数据集对象
    dataset_info = DatasetInfo(dataset_name, dataset_seed, dataset_size)

    msg = "From Dataset {} (based seed {}) selected {} sample(s)".format(dataset_name, dataset_seed, dataset_size)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    return dataset_info


def dataset_seed_handler(dataset_seed=None):
    if dataset_seed is None:
        return random.Random(batch_manager.batch_token).randint(10000000000000000, 100000000000000000)
    else:
        return dataset_seed