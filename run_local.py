import random

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager

# 模型
# VTI
from CANARY_SEFI.core.exec_local.security_evaluation import SecurityEvaluation
from Model.Vision_Transformer import sefi_component as vision_transformer_model
SEFI_component_manager.add(vision_transformer_model)
# Alexnet
from Model.Alexnet import sefi_component as alexnet_model
SEFI_component_manager.add(alexnet_model)
# VGG16
from Model.VGG16 import sefi_component as vgg_16_model
SEFI_component_manager.add(vgg_16_model)

# 攻击方案
# MIM
from Attack_Method.white_box_adv.MI_FGSM import sefi_component as mim_attacker
SEFI_component_manager.add(mim_attacker)

# 数据集
# IMAGENET2012
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset
SEFI_component_manager.add(imgnet2012_dataset)

if __name__ == "__main__":
    # SEFI_component_manager.debug()
    dataset = "ILSVRC-2012"
    attacker_list = ["MI-FGSM"]
    attacker_config = {
        "MI-FGSM": {
            "T": 1000,  # 迭代攻击轮数
            "epsilon": 0.1,  # 以无穷范数作为约束，设置最大值
            "pixel_min": -3,  # 像素值的下限
            "pixel_max": 3,  # 像素值的上限
            "alpha": 6 / 25,  # 每一轮迭代攻击的步长
            "attacktype": 'untargeted'  # 攻击类型：靶向 or 非靶向
        }
    }
    # model_list = ["Alexnet", "VGG-16"]
    model_list = ["Alexnet"]
    model_config = {
        "Alexnet": {},
        "VGG-16": {}
    }
    img_proc_config = {
        "Alexnet": {},
        "VGG-16": {}
    }
    dataset_size = 5

    explore_perturbation_config = {
        "MI-FGSM": {
            "upper_bound": 0.2,
            "lower_bound": 0,
            "step": 0.01,
            "dataset_size": None
        }
    }

    dataset_seed = random.randint(1000000000000, 10000000000000)

    security_evaluation = SecurityEvaluation()
    security_evaluation.full_adv_transfer_test = True
    security_evaluation.full_security_test(dataset, dataset_size, dataset_seed, attacker_list, attacker_config, model_list, model_config, img_proc_config)
    security_evaluation.explore_attack_perturbation_test(dataset, dataset_size, dataset_seed, attacker_list, attacker_config, model_list, model_config, img_proc_config, explore_perturbation_config)
