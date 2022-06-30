import random
import string

from colorama import Fore, Style

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.batch_flag import batch_flag
from CANARY_SEFI.core.exec_local.AEs_inference import AEs_inference
from CANARY_SEFI.core.exec_local.clear_inference import clear_inference
from CANARY_SEFI.core.exec_local.make_AEs import make_AEs
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo, DatasetType
from CANARY_SEFI.evaluator.logger.attack_logger import find_batch_adv_log
from CANARY_SEFI.evaluator.logger.dataset_logger import add_dataset_log
from CANARY_SEFI.evaluator.logger.db_logger import log

class SecurityEvaluation:
    def __init__(self):
        self.full_adv_transfer_test = False

    def model_capability_evaluation(self, dataset_info, model_list, model_config, img_proc_config):
        # 确定模型基线
        for model_name in model_list:
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            # 模型基线测试
            print(Fore.GREEN + "---->> [ BATCH {} ] [ 模型测试基线确定 ] 在模型 {} 上运行模型能力评估".format(batch_flag.batch_id, model_name))
            print(Style.RESET_ALL)
            clear_inference(dataset_info, model_name, model_args, img_proc_args)

    def adv_img_build_and_evaluation(self, dataset_info, attacker_list, attacker_config, model_list, model_config, img_proc_config):
        # 生成对抗样本
        for atk_name in attacker_list:
            print(Fore.GREEN + "---->> [  LIST  ] 攻击方案队列 {} ".format(attacker_list))
            print(Fore.GREEN + "---->> [ SELECT ] 攻击方案选定 {} ".format(atk_name))
            atk_args = attacker_config.get(atk_name, {})

            for model_name in model_list:
                print(Fore.GREEN + "---->> [  LIST  ] 待攻击模型队列 {} ".format(model_list))
                print(Fore.GREEN + "---->> [ SELECT ] 待攻击模型选定 {} ".format(model_name))
                model_args = model_config.get(model_name, {})
                img_proc_args = img_proc_config.get(model_name, {})

                print(Fore.GREEN + "---->> [ BATCH {} ] [ 生成对抗样本 ] 基于攻击方法 {} 在模型 {} 上生成上述样本的对抗样本并运行对抗样本质量评估".format(batch_flag.batch_id, atk_name, model_name))
                print(Style.RESET_ALL)
                make_AEs(dataset_info, atk_name, atk_args, model_name, model_args, img_proc_args)

    def attack_capability_test(self, batch_id, model_list, model_config, img_proc_config):
        # 验证攻击效果
        # 获取当前批次全部攻击图片目录
        all_adv_log = find_batch_adv_log(batch_id)
        for model_name in model_list:
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            print(Fore.GREEN + "---->> [ BATCH {} ] [ 模型攻击测试 ] 基于对抗样本在模型 {} 上运行模型能力评估".format(batch_id, model_name))
            print(Style.RESET_ALL)

            adv_img_cursor_list = []
            for adv_log in all_adv_log:
                if self.full_adv_transfer_test:
                    adv_img_cursor_list.append(adv_log[0])
                elif adv_log[3] == model_name:
                    adv_img_cursor_list.append(adv_log[0])

            adv_dataset_info = DatasetInfo(None, None, None, adv_img_cursor_list)
            adv_dataset_info.dataset_type = DatasetType.ADVERSARIAL_EXAMPLE

            AEs_inference(adv_dataset_info, model_name, model_args, img_proc_args)

    def attack_capability_evaluation(self, batch_id):
        print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 3  攻击测试结果分析 ] 基于攻击前后模型能力评估结果分析攻击效果".format(batch_id))
        print(Style.RESET_ALL)
        # todo:等待完成

    def full_security_test(self, dataset_name, dataset_size, dataset_seed, attacker_list, attacker_config, model_list, model_config, img_proc_config):
        print_logo()

        batch_flag.new_batch()
        print(Fore.GREEN + "---->> [  SEFI 开始测试 批次 {}  ] <<----".format(batch_flag.batch_id))

        # 数据集选定
        print(Fore.GREEN + "---->> [ STEP 0  测试数据集 ] 从数据集 {} (根据种子{}) 选定 {} 张样本在模型".format(dataset_name, dataset_seed, dataset_size))
        dataset_log_id = add_dataset_log(dataset_name, dataset_seed, dataset_size)

        # 构建数据集对象
        dataset_info = DatasetInfo(dataset_name, dataset_seed, dataset_size)
        dataset_info.dataset_log_id = dataset_log_id

        # 确定模型基线
        print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 0  模型测试基线确定 ] <<----".format(batch_flag.batch_id))
        self.model_capability_evaluation(dataset_info, model_list, model_config, img_proc_config)

        # 生成对抗样本
        print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 1  生成对抗样本    ] <<----".format(batch_flag.batch_id))
        self.adv_img_build_and_evaluation(dataset_info, attacker_list, attacker_config, model_list, model_config, img_proc_config)

        # 生成对抗样本
        print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 2  模型攻击测试    ] <<----".format(batch_flag.batch_id))
        self.attack_capability_test(batch_flag.batch_id, model_list, model_config, img_proc_config)

        # 攻击测试结果分析
        print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 3  攻击测试结果分析 ] <<----".format(batch_flag.batch_id))
        self.attack_capability_evaluation(batch_flag.batch_id)
        log.finish()