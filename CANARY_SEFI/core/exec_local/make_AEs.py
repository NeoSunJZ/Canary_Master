import random
import string

import torch
from tqdm import tqdm

from CANARY_SEFI.core.function.attacker_function import adv_attack_4_img_batch


def make_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args):
    with tqdm(total=dataset_info.dataset_size, desc="对抗样本生成进度", ncols=80) as bar:
        def each_img_finish_callback(img, adv_result):
            tqdm.write("[CUDA-REPORT] 正在释放缓存")
            torch.cuda.empty_cache()
            tqdm.write("[CUDA-REPORT] 当前CUDA显存使用量:\n{}".format(torch.cuda.memory_summary()))
            bar.update(1)

        adv_attack_4_img_batch(atk_name, atk_args, model_name, model_args, img_proc_args, dataset_info,
                               each_img_finish_callback)
