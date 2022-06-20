import random
import string
from tqdm import tqdm

from CANARY_SEFI.core.function.attacker_function import adv_attack_4_img_batch


def make_AEs(batch_token, dataset_name, dataset_size, atk_name, atk_args, model_name, model_args, img_proc_args):
    with tqdm(total=dataset_size, desc="对抗样本生成进度", ncols=80) as bar:
        def each_img_finish_callback(img, adv_result):
            bar.update(1)
        adv_attack_4_img_batch(batch_token, atk_name, atk_args, model_name, model_args, img_proc_args, dataset_name,
                                dataset_size, None, each_img_finish_callback)
