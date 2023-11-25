import random

from colorama import Fore

from canary_sefi.task_manager import task_manager
from canary_sefi.core.function.enum.step_enum import Step
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.entity.dataset_info_entity import DatasetInfo, DatasetType


def init_dataset(dataset_name, dataset_size, dataset_seed=None, dataset_path=None, dataset_folder=None,
                 dataset_type=DatasetType.NORMAL, n_classes=None, is_gray=False, is_fast_test=False):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.INIT, is_first=True)

    msg = "SEFI Initialization"
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    dataset_seed = dataset_seed_handler(dataset_seed)
    # 构建数据集对象
    dataset_extra_info = {
        "n_classes": n_classes,
        "is_gray": is_gray,
    }
    if dataset_path is not None:
        dataset_extra_info["path"] = dataset_path
    if dataset_folder is not None:
        dataset_extra_info["folder"] = dataset_folder

    dataset_extra_info["is_fast_test"] = is_fast_test
    dataset_info = DatasetInfo(dataset_name, dataset_extra_info, dataset_type, dataset_seed, dataset_size, img_cursor_list=None)

    msg = "From Dataset {} (based seed {}) selected {} sample(s)".format(dataset_name, dataset_seed, dataset_size)
    reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

    return dataset_info


def dataset_seed_handler(dataset_seed=None):
    if dataset_seed is None:
        return random.Random(task_manager.task_token).randint(10000000000000000, 100000000000000000)
    else:
        return dataset_seed