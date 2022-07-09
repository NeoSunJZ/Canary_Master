import random
import string

import torch
from tqdm import tqdm

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.function.attacker_function import adv_attack_4_img_batch


def build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args):
    with tqdm(total=dataset_info.dataset_size, desc="对抗样本生成进度", ncols=80) as bar:
        def each_img_finish_callback(img, adv_result):
            tqdm.write("[CUDA-REPORT] 正在释放缓存")
            torch.cuda.empty_cache()
            tqdm.write("[CUDA-REPORT] 当前CUDA显存使用量:{}".format(torch.cuda.memory_allocated()))
            bar.update(1)

        adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info,
                               each_img_finish_callback)
        tqdm.write("[CUDA-REPORT] 正在释放缓存")
        torch.cuda.empty_cache()
        tqdm.write("[CUDA-REPORT] 当前CUDA显存使用量详情:\n{}".format(torch.cuda.memory_summary()))
        bar.update(1)


def explore_perturbation(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args,
                         explore_perturbation_args=None):
    if explore_perturbation_args is None:
        explore_perturbation_args = {"upper_bound": 0.2, "lower_bound": 0, "step": 0.01, "dataset_size": None}

    atk_component = SEFI_component_manager.attack_method_list.get(atk_name)
    perturbation_budget_var_name = atk_component.get('attacker_class').get('perturbation_budget_var_name')

    if explore_perturbation_args['dataset_size'] is not None:
        dataset_info.dataset_size = explore_perturbation_args['dataset_size']

    lower_bound = explore_perturbation_args['lower_bound']
    upper_bound = explore_perturbation_args['upper_bound']
    step = explore_perturbation_args['step']

    with tqdm(total=(upper_bound-lower_bound)/step, desc="扰动探索测试进度", ncols=80) as bar:

        now_perturbation = lower_bound
        while now_perturbation < upper_bound:
            # 扰动预算前进步长
            now_perturbation += step
            atk_args[perturbation_budget_var_name] = now_perturbation

            build_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args)

            bar.update(1)
