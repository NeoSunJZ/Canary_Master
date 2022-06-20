import random
import string

from colorama import Fore, Style

from CANARY_SEFI.copyright import print_logo
from CANARY_SEFI.core.exec_local.AEs_inference import AEs_inference
from CANARY_SEFI.core.exec_local.clear_inference import clear_inference
from CANARY_SEFI.core.exec_local.make_AEs import make_AEs


def full_security_test(dataset_name, dataset_size, attacker_list, attacker_config, model_list, model_config, img_proc_config):
    print_logo()
    print(Fore.GREEN + "---->> [ SEFI 开始测试 ]")
    for atk_name in attacker_list:
        print("---->> [  LIST  ] 攻击方案队列 {} ".format(attacker_list))
        print("---->> [ SELECT ] 攻击方案选定 {} ".format(atk_name))
        atk_args = attacker_config.get(atk_name, {})
        for model_name in model_list:

            print(Fore.GREEN + "---->> [  LIST  ] 待攻击模型队列 {} ".format(model_list))
            print(Fore.GREEN + "---->> [ SELECT ] 待攻击模型选定 {} ".format(model_name))
            print(Style.RESET_ALL)
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            batch_token = ''.join(random.sample(string.ascii_letters + string.digits, 8))

            print(Fore.GREEN + "---->> [ BATCH {} ] 选定数据集{} 选定样本量{} 选定模型{} 选定攻击方法{}".format(batch_token, dataset_name, dataset_size, model_name, atk_name))
            print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 1  模型测试基线确定 ] 从数据集 {} 选定 {} 张样本在模型 {} 上运行模型能力评估".format(batch_token, dataset_name, dataset_size, model_name))
            print(Style.RESET_ALL)
            clear_inference(batch_token, dataset_name, dataset_size, model_name, model_args, img_proc_args)

            print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 2  生成对抗样本    ] 基于攻击方法 {} 在模型 {} 上生成上述样本的对抗样本并运行对抗样本质量评估".format(batch_token, atk_name, model_name))
            print(Style.RESET_ALL)
            make_AEs(batch_token, dataset_name, dataset_size, atk_name, atk_args, model_name, model_args, img_proc_args)

            print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 3  模型攻击测试    ] 基于上述对抗样本在模型 {} 上运行模型能力评估".format(batch_token, model_name))
            print(Style.RESET_ALL)
            AEs_inference(batch_token, dataset_size, model_name, model_args, img_proc_args)

            print(Fore.GREEN + "---->> [ BATCH {} ] [ STEP 4  攻击测试结果分析 ] 基于攻击前后模型能力评估结果分析攻击效果".format(batch_token))
            print(Style.RESET_ALL)
