import random
import string

import torch
from colorama import Fore, Style
from tqdm import tqdm

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.exec_local.AEs_inference import AEs_inference
from CANARY_SEFI.core.exec_local.clear_inference import clear_inference
from CANARY_SEFI.core.exec_local.make_AEs import make_AEs
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.evaluator.analyzer.test_analyzer import model_capability_evaluation, attack_capability_evaluation
from CANARY_SEFI.evaluator.logger.adv_logger import find_batch_adv_log
from CANARY_SEFI.evaluator.logger.attack_logger import find_attack_log
from CANARY_SEFI.evaluator.logger.dataset_logger import add_dataset_log
from CANARY_SEFI.evaluator.logger.db_logger import log

class SecurityEvaluation:
    def __init__(self):
        self.full_adv_transfer_test = False

    def model_capability_test(self, dataset_info, model_list, model_config, img_proc_config):
        # 确定模型基线
        for model_name in model_list:
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            # 模型基线测试
            tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ 模型测试基线确定 ] 在模型 {} 上运行模型能力评估".format(batch_flag.batch_id, model_name))
            tqdm.write(Style.RESET_ALL)
            clear_inference(dataset_info, model_name, model_args, img_proc_args)

            torch.cuda.empty_cache()
            tqdm.write(" [ BATCH {} ] [ 模型测试基线确定 ] [CUDA-REPORT] 已清理CUDA缓存，当前CUDA显存使用量:\n{}".format(batch_flag.batch_id,torch.cuda.memory_summary()))

    def adv_img_build_and_evaluation(self, dataset_info, attacker_list, attacker_config, model_list, model_config, img_proc_config):
        # 生成对抗样本
        for atk_name in attacker_list:
            tqdm.write(Fore.GREEN + "---->> [  LIST  ] 攻击方案队列 {} ".format(attacker_list))
            tqdm.write(Fore.GREEN + "---->> [ SELECT ] 攻击方案选定 {} ".format(atk_name))
            atk_args = attacker_config.get(atk_name, {})

            for model_name in model_list:
                tqdm.write(Fore.GREEN + "---->> [  LIST  ] 待攻击模型队列 {} ".format(model_list))
                tqdm.write(Fore.GREEN + "---->> [ SELECT ] 待攻击模型选定 {} ".format(model_name))
                model_args = model_config.get(model_name, {})
                img_proc_args = img_proc_config.get(model_name, {})

                tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ 生成对抗样本 ] 基于攻击方法 {} 在模型 {} 上生成上述样本的对抗样本并运行对抗样本质量评估".format(batch_flag.batch_id, atk_name, model_name))
                tqdm.write(Style.RESET_ALL)
                make_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args)

                torch.cuda.empty_cache()
                tqdm.write(" [ BATCH {} ] [ 生成对抗样本 ] [CUDA-REPORT] 已清理CUDA缓存，当前CUDA显存使用量:\n{}".format(batch_flag.batch_id,torch.cuda.memory_summary()))

    def attack_capability_test(self, batch_id, model_list, model_config, img_proc_config):
        # 验证攻击效果
        # 获取当前批次全部攻击图片目录
        all_adv_log = find_batch_adv_log(batch_id)
        for model_name in model_list:
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ 模型攻击测试 ] 基于对抗样本在模型 {} 上运行模型能力评估".format(batch_id, model_name))
            tqdm.write(Style.RESET_ALL)

            adv_img_cursor_list = []
            for adv_log in all_adv_log:
                if self.full_adv_transfer_test:
                    adv_img_cursor_list.append(adv_log[0])
                else:
                    attack_log = find_attack_log(adv_log[2])[0]
                    if attack_log[3] == model_name:
                        adv_img_cursor_list.append(adv_log[0])

            adv_dataset_info = DatasetInfo(None, None, None, adv_img_cursor_list)
            adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE

            AEs_inference(adv_dataset_info, model_name, model_args, img_proc_args)

            torch.cuda.empty_cache()
            tqdm.write(" [ BATCH {} ] [ 模型攻击测试 ] [CUDA-REPORT] 已清理CUDA缓存，当前CUDA显存使用量:\n{}".format(batch_flag.batch_id, torch.cuda.memory_summary()))

    def full_security_test(self, dataset_name, dataset_size, dataset_seed, attacker_list, attacker_config, model_list, model_config, img_proc_config):
        print_logo()

        batch_flag.new_batch()
        tqdm.write(Fore.GREEN + "---->> [  SEFI 开始测试 批次 {}  ] <<----".format(batch_flag.batch_id))

        # 数据集选定
        tqdm.write(Fore.GREEN + "---->> [ STEP 0  测试数据集 ] 从数据集 {} (根据种子{}) 选定 {} 张样本在模型".format(dataset_name, dataset_seed, dataset_size))
        dataset_log_id = add_dataset_log(dataset_name, dataset_seed, dataset_size)

        # 构建数据集对象
        dataset_info = DatasetInfo(dataset_name, dataset_seed, dataset_size)
        dataset_info.dataset_log_id = dataset_log_id

        # 模型基线能力评估
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 0  模型基线能力评估 ] <<----".format(batch_flag.batch_id))
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 0  正在运行测试 ] <<----".format(batch_flag.batch_id))
        self.model_capability_test(dataset_info, model_list, model_config, img_proc_config)
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 0  分析测试结果 ] <<----".format(batch_flag.batch_id))
        model_capability_evaluation(batch_flag.batch_id)
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 0  模型基线能力评估完成 ] <<----".format(batch_flag.batch_id))


        # 生成对抗样本
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 1  生成对抗样本    ] <<----".format(batch_flag.batch_id))
        self.adv_img_build_and_evaluation(dataset_info, attacker_list, attacker_config, model_list, model_config, img_proc_config)

        # 生成对抗样本
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 2  模型攻击测试    ] <<----".format(batch_flag.batch_id))
        self.attack_capability_test(batch_flag.batch_id, model_list, model_config, img_proc_config)

        # 攻击测试结果分析
        tqdm.write(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 3  攻击测试结果分析 ] <<----".format(batch_flag.batch_id))
        attack_capability_evaluation(batch_flag.batch_id)
        log.finish()
