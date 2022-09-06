from colorama import Fore

from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo
from CANARY_SEFI.evaluator.logger.dataset_logger import add_dataset_log


def init_dataset(dataset_name, dataset_size, dataset_seed):
    # 标记当前步骤
    global_system_log.set_step("INIT", is_first=True)

    msg = "SEFI任务初始化"
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    msg = "从数据集 {} (根据种子{}) 选定 {} 张样本在模型".format(dataset_name, dataset_seed, dataset_size)
    reporter.console_log(msg, Fore.GREEN, show_batch=True, show_step_sequence=True)

    dataset_log_id = add_dataset_log(dataset_name, dataset_seed, dataset_size)

    # 构建数据集对象
    dataset_info = DatasetInfo(dataset_name, dataset_seed, dataset_size)
    dataset_info.dataset_log_id = dataset_log_id

    return dataset_info