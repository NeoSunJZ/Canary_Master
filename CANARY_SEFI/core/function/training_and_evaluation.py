from colorama import Fore

from CANARY_SEFI.core.function.adversarial_training import adv_training
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.enum.step_enum import Step
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.core.function.helper.batch_list_iterator import BatchListIterator


def adv_defense_training(dataset_info, defense_list, defense_config, model_config, img_proc_config):
    # 标记当前步骤
    task_manager.sys_log_logger.set_step(Step.DEFENSE_ADVERSARIAL_TRAINING)

    def function(defense_name, defense_args, model_name, model_args, img_proc_args, run_device):
        msg = "[Device {}] Adversarial Training With Defense Method ({}) on Model ({}).".format(run_device,
                                                                                                  defense_name,
                                                                                                  model_name)
        reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)
        adv_training(dataset_info, defense_name, defense_args, model_name, model_args, img_proc_args,
                     run_device=run_device)

    BatchListIterator.defense_list_iterator(defense_list, defense_config, model_config, img_proc_config, function)
