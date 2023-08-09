import torch
from colorama import Fore
from tqdm import tqdm

from canary_sefi.task_manager import task_manager
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.function.basic.attacker_function import adv_attack_4_img_batch
from canary_sefi.core.function.basic.train_function import adv_defense_4_img_batch
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.core.function.helper.recovery import global_recovery
from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


def adv_training(dataset_info, defense_name, defense_args, model_name, model_args, img_proc_args, run_device=None):
    with tqdm(total=defense_args["epochs"], desc="Adversarial Training Progress", ncols=120) as bar:
        def each_epoch_finish_callback(epoch):
            check_cuda_memory_alloc_status(empty_cache=True)
            task_manager.sys_log_logger.update_completed_num(1)
            bar.update(1)

        participant = "{}({})".format(defense_name, model_name)

        is_skip, completed_num = global_recovery.check_skip(participant)
        if is_skip:
            return None

        adv_defense_4_img_batch(defense_name, defense_args, model_name, model_args, img_proc_args, dataset_info,
                                each_epoch_finish_callback=each_epoch_finish_callback, run_device=run_device)
        check_cuda_memory_alloc_status(empty_cache=True)
        bar.update(1)
