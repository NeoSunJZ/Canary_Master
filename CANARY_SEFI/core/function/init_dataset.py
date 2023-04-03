import random

from colorama import Fore

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo


def init_dataset(dataset_name, dataset_size, dataset_seed=None, is_train=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.INIT, is_first=True)

    msg = "SEFI Initialization"
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    dataset_seed = dataset_seed_handler(dataset_seed)
    # 构建数据集对象
    dataset_info = DatasetInfo(dataset_name=dataset_name, dataset_seed=dataset_seed, dataset_size=dataset_size, is_train=is_train)
    dataset_info.n_classes = config_manager.config.get("dataset", {}).get(dataset_info.dataset_name, {}).get("n_classes", None)

    msg = "From Dataset {} (based seed {}) selected {} sample(s)".format(dataset_name, dataset_seed, dataset_size)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    return dataset_info


def dataset_seed_handler(dataset_seed=None):
    if dataset_seed is None:
        return random.Random(task_manager.task_token).randint(10000000000000000, 100000000000000000)
    else:
        return dataset_seed