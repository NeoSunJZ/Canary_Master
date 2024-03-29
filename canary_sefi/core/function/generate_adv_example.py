import torch
from colorama import Fore
from tqdm import tqdm

from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.function.basic.attacker_function import adv_attack_4_img_batch
from canary_sefi.core.function.helper.realtime_reporter import reporter
from canary_sefi.core.function.helper.recovery import global_recovery
from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


def build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args, atk_batch_config, atk_perturbation_budget=None, run_device=None):
    with tqdm(total=dataset_info.dataset_size, desc="Adv-example Generating progress", ncols=120) as bar:
        def each_img_finish_callback(img, adv_result):
            check_cuda_memory_alloc_status(empty_cache=True)
            bar.update(1)

        participant = "{}({})".format(atk_name, model_name)
        if atk_perturbation_budget != "None" and atk_perturbation_budget is not None:
            participant += "(e-{})".format(str(round(float(atk_perturbation_budget), 5)))
        is_skip, completed_num = global_recovery.check_skip(participant)
        if is_skip:
            return None
        batch_size = atk_batch_config.get(atk_name, {}).get(model_name, 1)
        adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info,
                               each_img_finish_callback=each_img_finish_callback,
                               batch_size=batch_size,
                               completed_num=completed_num,
                               run_device=run_device)
        check_cuda_memory_alloc_status(empty_cache=True)
        bar.update(1)


def build_AEs_with_perturbation_increment(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                                          atk_batch_config, perturbation_increment_args=None, run_device=None):
    if perturbation_increment_args is None:
        perturbation_increment_args = {"upper_bound": 0.2, "lower_bound": 0, "step": 0.01}

    atk_component = SEFI_component_manager.attack_method_list.get(atk_name)
    perturbation_budget_var_name = atk_component.get('attacker_class').get('perturbation_budget_var_name')

    lower_bound = perturbation_increment_args['lower_bound']
    upper_bound = perturbation_increment_args['upper_bound']
    step = perturbation_increment_args['step']

    with tqdm(total=(upper_bound-lower_bound)/step, desc="Perturbation Increasing progress", ncols=120) as bar:

        now_perturbation = lower_bound
        while now_perturbation < upper_bound:
            # 扰动预算前进步长
            now_perturbation += step

            msg = "Generating Adv Example By Attack Method {} on(base) Model {}(Now perturbation:{}).".format(atk_name, model_name, now_perturbation)
            reporter.console_log(msg, Fore.GREEN, show_task=True, show_step_sequence=True)

            atk_args[perturbation_budget_var_name] = now_perturbation
            build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                      atk_batch_config, now_perturbation, run_device)
            bar.update(1)
