from CANARY_SEFI.batch_manager import batch_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager

# 模型
# VTI
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from Model.Vision_Transformer import sefi_component as vision_transformer_model

SEFI_component_manager.add(vision_transformer_model)
# Alexnet
from Model.Alexnet_ImageNet import sefi_component as alexnet_model

SEFI_component_manager.add(alexnet_model)
# VGG16
from Model.VGG16_ImageNet import sefi_component as vgg_16_model

SEFI_component_manager.add(vgg_16_model)

# 攻击方案
# CW
from Attack_Method.white_box_adv.CW import sefi_component as cw_attacker

SEFI_component_manager.add(cw_attacker)
# MI-FGSM
from Attack_Method.white_box_adv.MI_FGSM import sefi_component as mi_fgsm_attacker

SEFI_component_manager.add(mi_fgsm_attacker)

# 数据集
# IMAGENET2012
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset

SEFI_component_manager.add(imgnet2012_dataset)

if __name__ == "__main__":

    # config = {"dataset_size": 2,"dataset": "ILSVRC-2012",
    #           "model_list": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"]
    #           "model_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}}, "img_proc_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
    #           "attacker_list": {"CW": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"],
    #                             "MI-FGSM": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"]},
    #           "transfer_attack_test_mode": "SELF_CROSS",
    #           "transfer_attack_test_on_model_list": {},
    #           "attacker_config": {
    #               "CW": {"classes": 1000, "lr": 0.005, "confidence": 0, "clip_min": -3, "clip_max": 3,
    #                      "initial_const": 0.01,
    #                      "binary_search_steps": 5, "max_iterations": 1000, "attack_type": "UNTARGETED", "tlabel": None},
    #               "MI-FGSM": {"alpha": 0.005, "epsilon": 0.1, "pixel_min": -3, "pixel_max": 3, "T": 1000,
    #                           "attack_type": "UNTARGETED", "tlabel": None}}
    #           }
    # security_evaluation = SecurityEvaluation(config)
    # security_evaluation.attack_full_test(use_img_file=True, use_raw_nparray_data=True)

    config = {"dataset_size": 150, "dataset": "ILSVRC-2012",
              "model_list": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"],
              "model_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
              "img_proc_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
              "attacker_list": {"MI-FGSM": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"]},
              "attacker_config": {
                  "MI-FGSM": {"alpha": 0.002, "epsilon": 0, "pixel_min": -3, "pixel_max": 3, "T": 1000,
                              "attack_type": "UNTARGETED", "tlabel": None}
              },
              "perturbation_increment_config": {
                  "MI-FGSM": {
                      "upper_bound": 0.02, "lower_bound": 0.008, "step": 0.001,
                  }
              }
              }
    batch_manager.init_batch(show_logo=True)
    security_evaluation = SecurityEvaluation(config)
    security_evaluation.attack_perturbation_increment_test(use_img_file=True, use_raw_nparray_data=True)

    # config_explore_perturbation = {"dataset_size": 150, "model_list": ["Alexnet", "VGG-16"], "dataset": "ILSVRC-2012",
    #                                "model_config": {"Alexnet": {}, "VGG-16": {}},
    #                                "img_proc_config": {"Alexnet": {}, "VGG-16": {}},
    #                                "attacker_list": {"MI-FGSM": [ "VGG-16","Alexnet"]},
    #                                "attacker_config": {
    #                                    "MI-FGSM": {"alpha": 0.00005, "epsilon": 0, "pixel_min": -3, "pixel_max": 3,
    #                                                "T": 1000,
    #                                                "attack_type": "UNTARGETED", "tlabel": None}},
    #                                "explore_perturbation_config": {
    #                                    "MI-FGSM": {
    #                                        "upper_bound": 0.0095,
    #                                        "lower_bound": 0.0065,
    #                                        "step": 0.0002,
    #                                    }
    #                                }
    #                                }
    # config_explore_perturbation = {"dataset_size": 100, "model_list": ["Alexnet"], "dataset": "ILSVRC-2012",
    #                                "model_config": {"Alexnet": {}},
    #                                "img_proc_config": {"Alexnet": {}},
    #                                "attacker_list": {"MI-FGSM": ["Alexnet"]},
    #                                "attacker_config": {
    #                                    "MI-FGSM": {"alpha": 0.002, "epsilon": 0, "pixel_min": -3, "pixel_max": 3,
    #                                                "T": 1000,
    #                                                "attack_type": "UNTARGETED", "tlabel": None}},
    #                                "explore_perturbation_config": {
    #                                    "MI-FGSM": {
    #                                        "upper_bound": 0.02,
    #                                        "lower_bound": 0,
    #                                        "step": 0.001,
    #                                    }
    #                                }
    #                                }

    # model_security_synthetical_capability_evaluation("SCPzbXIq", config.get('model_list'))

    # global_recovery.start_recovery_mode("yfjnmocE")
    # security_evaluation = SecurityEvaluation(config.get('dataset'), config.get('dataset_size'),
    #                                          config.get('dataset_seed', None))
    # security_evaluation.attack_full_test(config.get('attacker_list'), config.get('attacker_config'),
    #                                      config.get('model_list'), config.get('model_config'),
    #                                      config.get('img_proc_config'),
    #                                      config.get('transfer_attack_test_mode'),
    #                                      config.get('transfer_attack_test_on_model_list', None))

    # security_evaluation.only_build_adv(config.get('attacker_list'), config.get('attacker_config'),
    #                                    config.get('model_config'), config.get('img_proc_config'))
    # global_recovery.start_recovery_mode("yfjnmocE")

    # ori_info = init_dataset(config_explore_perturbation.get('dataset'), config_explore_perturbation.get('dataset_size'),
    #                                          config_explore_perturbation.get('dataset_seed', None))
    # img_ori = dataset_single_image_reader(ori_info, 0)
    # img_adv_1 = adv_dataset_single_image_reader(1)
    # img_adv_2 = adv_dataset_single_image_reader(2101)

    # security_evaluation = SecurityEvaluation(config_explore_perturbation.get('dataset'), config_explore_perturbation.get('dataset_size'),
    #                                          config_explore_perturbation.get('dataset_seed', None))
    # security_evaluation.explore_attack_perturbation_test(config_explore_perturbation.get('attacker_list'),
    #                                                      config_explore_perturbation.get('attacker_config'),
    #                                                      config_explore_perturbation.get('model_config'),
    #                                                      config_explore_perturbation.get('img_proc_config'),
    #                                                      config_explore_perturbation.get('explore_perturbation_config'))
