import Attack_Method.black_box_adv.BA
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
# 模型
# Alexnet
from Model.ImageNet.Alexnet_ImageNet import sefi_component as alexnet_model
# ConvNext
from Model.ImageNet.ConvNeXt_ImageNet import sefi_component as convnext_model
# DenseNet
from Model.ImageNet.DenseNet_ImageNet import sefi_component as densenet_model_imagenet
from Model.CIFAR10.DenseNet.DenseNet_CIFAR10 import sefi_component as densenet_model_cifar10
# EfficientNet
from Model.ImageNet.EfficientNet_ImageNet import sefi_component as efficientnet_model
# EfficientNetV2
from Model.ImageNet.EfficientNetV2_ImageNet import sefi_component as efficientnetV2_model
# GoogLeNet
from Model.ImageNet.GoogLeNet_ImageNet import sefi_component as googlenet_model
# InceptionV3
from Model.ImageNet.InceptionV3_ImageNet import sefi_component as inceptionV3_model
# MNASNet
from Model.ImageNet.MNASNet_ImageNet import sefi_component as mnasnet_model
# MobileNetV2
from Model.ImageNet.MobileNetV2_ImageNet import sefi_component as mobilenetv2_model
# MobileNetV3
from Model.ImageNet.MobileNetV3_ImageNet import sefi_component as mobilenetv3_model
# RegNet
from Model.ImageNet.RegNet_ImageNet import sefi_component as regnet_model
# ResNet
from Model.ImageNet.ResNet_ImageNet import sefi_component as resnet_model
# ResNeXt
from Model.ImageNet.ResNeXt_ImageNet import sefi_component as resnext_model
# ShuffleNetV2
from Model.ImageNet.ShuffleNetV2_ImageNet import sefi_component as shufflenetV2_model
# SqueezeNet
from Model.ImageNet.SqueezeNet_ImageNet import sefi_component as squeezenet_model
# SwinTransformer
from Model.ImageNet.SwinTransformer_ImageNet import sefi_component as swintransformer_model
# VGG
from Model.ImageNet.VGG_ImageNet import sefi_component as vgg_model
# ViT
from Model.ImageNet.VisionTransformer_ImageNet import sefi_component as vit_model
# WideResNet
from Model.ImageNet.WideResNet_ImageNet import sefi_component as wideresnet_model

SEFI_component_manager.add(alexnet_model)
SEFI_component_manager.add(convnext_model)
SEFI_component_manager.add(densenet_model_imagenet)
SEFI_component_manager.add(densenet_model_cifar10)
SEFI_component_manager.add(efficientnet_model)
SEFI_component_manager.add(efficientnetV2_model)
SEFI_component_manager.add(googlenet_model)
SEFI_component_manager.add(inceptionV3_model)
SEFI_component_manager.add(mnasnet_model)
SEFI_component_manager.add(mobilenetv2_model)
SEFI_component_manager.add(mobilenetv3_model)
SEFI_component_manager.add(regnet_model)
SEFI_component_manager.add(resnet_model)
SEFI_component_manager.add(resnext_model)
SEFI_component_manager.add(shufflenetV2_model)
SEFI_component_manager.add(squeezenet_model)
SEFI_component_manager.add(swintransformer_model)
SEFI_component_manager.add(vgg_model)
SEFI_component_manager.add(vit_model)
SEFI_component_manager.add(wideresnet_model)

from Model.ImageNet.common import sefi_component as common_imagenet
from Model.CIFAR10.common import sefi_component as common_cifar10
SEFI_component_manager.add(common_imagenet)
SEFI_component_manager.add(common_cifar10)

# 攻击方案
# CW
from Attack_Method.white_box_adv.CW import sefi_component as cw_attacker
# MI-FGSM
from Attack_Method.white_box_adv.MI_FGSM import sefi_component as mi_fgsm_attacker
# UAP
from Attack_Method.white_box_adv.UAP import sefi_component as uap_attacker
# DeepFool
from Attack_Method.white_box_adv.DeepFool import sefi_component as deepfool_attacker
# EAD
from Attack_Method.white_box_adv.EAD import sefi_component as ead_attacker
# SA
from Attack_Method.black_box_adv.SA import sefi_component as sa_attacker
# ADVGAN
from Attack_Method.black_box_adv.ADVGAN import sefi_component as advgan_attacker
# L_BFGS
from Attack_Method.white_box_adv.L_BFGS import sefi_component as lbfgs_attacker

SEFI_component_manager.add(cw_attacker)
SEFI_component_manager.add(mi_fgsm_attacker)
SEFI_component_manager.add(uap_attacker)
SEFI_component_manager.add(deepfool_attacker)
SEFI_component_manager.add(ead_attacker)
SEFI_component_manager.add(sa_attacker)
SEFI_component_manager.add(advgan_attacker)
SEFI_component_manager.add(lbfgs_attacker)

# 数据集
# IMAGENET2012
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset
from Dataset.CIFAR10.dataset_loader import sefi_component as cifar10_dataset
SEFI_component_manager.add(cifar10_dataset)
SEFI_component_manager.add(imgnet2012_dataset)

if __name__ == "__main__":
    config = {"dataset_size": 5, "dataset": "ILSVRC-2012",
              "dataset_seed": 40376958655838027,
              "model_list": [
                  "DenseNet(ImageNet)",
              ],
              "attacker_list": {
                  # "CW": [
                  #     "DenseNet(ImageNet)",
                  # ],
                  # "EAD": [
                  #     "DenseNet(ImageNet)",
                  # ],
                  # "SA": [
                  #     "DenseNet(ImageNet)",
                  # ],
                  # "ADVGAN": [
                  #     "DenseNet(ImageNet)",
                  # ],
                  "L_BFGS": [
                      "DenseNet(ImageNet)",
                  ],
              },
              "attacker_config": {
                  # "CW": {
                  #     "classes": 1000,
                  #     "run_device":'cuda',
                  #     "tlabel": None,
                  #     "attack_type": "UNTARGETED",
                  #     "clip_min": 0,
                  #     "clip_max": 1,
                  #     "lr": 2e-2,
                  #     "initial_const": 1e-2,
                  #     "binary_search_steps": 5,
                  #     "max_iterations": 250,
                  # },
                  # "EAD": {
                  #     "clip_min": -3,
                  #     "clip_max": 3,
                  #     "epsilon": 0.03,
                  #     "attack_type": 'UNTARGETED',
                  #     "tlabel": None,
                  #     "steps": 10,
                  #     "initial_const": 1e-3,
                  #     "binary_search_steps": 9,
                  #     "initial_stepsize": 1e-2,
                  #     "confidence": 0.0,
                  #     "regularization": 1e-2,
                  #     "decision_rule": 'EN',
                  #     "abort_early": True,
                  # },
                  # "SA": {
                  #     "clip_min": -3,
                  #     "clip_max": 3,
                  #     "epsilon": 0.03,
                  #     "attack_type": 'UNTARGETED',
                  #     "tlabel": None,
                  #     "max_translation": 3,
                  #     "max_rotation": 30,
                  #     "num_translations": 5,
                  #     "num_rotations": 5,
                  #     "grid_search": True,
                  #     "random_steps": 100,
                  # },
                  # "ADVGAN": {
                  #         "model_num_labels": 1000,
                  #         "image_nc": 3,
                  #         "output_nc": 3,
                  #         "box_min": 0,
                  #         "box_max": 1,
                  #         "lr": 0.001,
                  #         "epochs": 60,
                  # },
                  "L_BFGS": {
                          "bounds_min": 0.0,
                          "bounds_max": 1.0,
                          "epsilon": 0.01,
                          "steps": 10,
                          "attack_type": 'UNTARGETED',
                          "attack_target": None,
                  },
              },
              "inference_batch_config":{
                  "DenseNet(ImageNet)": 15,
              },
              "adv_example_generate_batch_config": {
                  # "CW": {
                  #     "DenseNet(ImageNet)":15,
                  # },
                  # "EAD": {
                  #     "DenseNet(ImageNet)":15,
                  # },
                  # "SA": {
                  #     "DenseNet(ImageNet)": 15,
                  # },
                  # "ADVGAN": {
                  #     "DenseNet(ImageNet)": 15,
                  # },
                  "L_BFGS": {
                      "DenseNet(ImageNet)": 15,
                  },
              }
              }
    task_manager.init_task(show_logo=True)
    security_evaluation = SecurityEvaluation(config)
    security_evaluation.adv_example_generate()

    # "ConvNext(ImageNet)",
    # "DenseNet(ImageNet)",
    # "EfficientNet(ImageNet)",
    # "EfficientNetV2(ImageNet)",
    # "GoogLeNet(ImageNet)",
    # "InceptionV3(ImageNet)",
    # "MNASNet(ImageNet)",
    # "MobileNetV2(ImageNet)",
    # "MobileNetV3(ImageNet)",
    # "RegNet(ImageNet)",
    # "ResNet(ImageNet)",
    # "ResNeXt(ImageNet)",
    # "ShuffleNetV2(ImageNet)",
    # "SqueezeNet(ImageNet)",
    # "SwinTransformer(ImageNet)",
    # "VGG(ImageNet)",
    # "ViT(ImageNet)",
    # "WideResNet(ImageNet)",

    # config = {"dataset_size": 1000, "dataset": "CIFAR-10",
    #           "dataset_seed": 40376958655838027,
    #           "model_list": [
    #               "DenseNet(CIFAR-10)",
    #           ],
    #           "attacker_list": {
    #               "CW": [
    #                   "DenseNet(CIFAR-10)",
    #               ],
    #           },
    #           "attacker_config": {
    #               "CW": {
    #                   "classes": 1000,
    #                   "tlabel": None,
    #                   "attack_type": "UNTARGETED",
    #                   "clip_min": 0,
    #                   "clip_max": 1,
    #                   "lr": 2e-2,
    #                   "initial_const": 1e-2,
    #                   "binary_search_steps": 5,
    #                   "max_iterations":250,
    #               }
    #           }}
    # batch_manager.init_batch(show_logo=True)
    # security_evaluation = SecurityEvaluation(config)
    # security_evaluation.adv_example_generate()

    # config = {"dataset_size": 2,"dataset": "ILSVRC-2012",
    #           "model_list": [
    #               "ConvNext(ImageNet)",
    #           ],
    #           "model_config": {"ConvNext(ImageNet)": {},},
    #           "img_proc_config": {"ConvNext(ImageNet)": {},},
    #           "attacker_list": {
    #               "MI-FGSM": [
    #                   "ConvNext(ImageNet)",
    #               ],
    #           },
    #           "transfer_attack_test_mode": "NOT",
    #           "transfer_attack_test_on_model_list": {},
    #           "attacker_config": {
    #               "MI-FGSM": {"alpha": 0.001, "epsilon": None, "pixel_min": 0, "pixel_max": 1, "T": 1000,
    #                           "attack_type": "UNTARGETED", "tlabel": None}
    #               # "UAP": {"num_classes":1000,
    #               #         "xi":10 / 255.0,
    #               #         "p":"l-inf",
    #               #         "overshoot":0.02,
    #               #         "delta":0.2,
    #               #         "max_iter_df":10
    #               #         }
    #               }
    #           }
    # config = {"dataset_size": 2,"dataset": "ILSVRC-2012",
    #           "model_list": [
    #               "Alexnet(ImageNet)",
    #           ],
    #           "model_config": {"Alexnet(ImageNet)": {},},
    #           "img_proc_config": {"Alexnet(ImageNet)": {},},
    #           "attacker_list": {
    #               "DeepFool": [
    #                   "Alexnet(ImageNet)",
    #               ],
    #           },
    #           "transfer_attack_test_mode": "NOT",
    #           "transfer_attack_test_on_model_list": {},
    #           "attacker_config": {
    #               "DeepFool": {
    #                   "pixel_min": 0,
    #                   "pixel_max": 1,
    #                   "num_classes":1000,
    #                   "p":"l-2",
    #                   "overshoot":0.02,
    #                   "max_iter":10
    #               }}
    #           }

    # config = {"dataset_size": 3,"dataset": "ILSVRC-2012",
    #           "model_list": [
    #               "Alexnet(ImageNet)",
    #               # "VGG-16(ImageNet)"
    #           ],
    #           "model_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
    #           "img_proc_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
    #           "attacker_list": {
    #               # "CW": [
    #               #     "Alexnet(ImageNet)",
    #               #     "VGG-16(ImageNet)"
    #               # ],
    #               "MI-FGSM": [
    #                   "Alexnet(ImageNet)",
    #                   # "VGG-16(ImageNet)"
    #               ],
    #           },
    #           # "transfer_attack_test_mode": "SELF_CROSS",
    #           "transfer_attack_test_mode": "NOT",
    #           "transfer_attack_test_on_model_list": {},
    #           "attacker_config": {
    #               # "CW": {"classes": 1000, "lr": 0.005, "confidence": 0, "clip_min": -3, "clip_max": 3,
    #               #        "initial_const": 0.01,
    #               #        "binary_search_steps": 5, "max_iterations": 1000, "attack_type": "UNTARGETED", "tlabel": None},
    #               "MI-FGSM": {"alpha": 0.005, "epsilon": 0.2, "pixel_min": -3, "pixel_max": 3, "T": 1000,
    #                           "attack_type": "UNTARGETED", "tlabel": None}
    #           }
    #           }
    # batch_manager.init_batch(show_logo=True)
    # security_evaluation = SecurityEvaluation(config)
    # security_evaluation.attack_full_test(use_img_file=True, use_raw_nparray_data=True)

    # config = {"dataset_size": 150, "dataset": "ILSVRC-2012",
    #           "model_list": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"],
    #           "model_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
    #           "img_proc_config": {"Alexnet(ImageNet)": {}, "VGG-16(ImageNet)": {}},
    #           "attacker_list": {"MI-FGSM": ["Alexnet(ImageNet)", "VGG-16(ImageNet)"]},
    #           "attacker_config": {
    #               "MI-FGSM": {"alpha": 0.002, "epsilon": 0, "pixel_min": -3, "pixel_max": 3, "T": 1000,
    #                           "attack_type": "UNTARGETED", "tlabel": None}
    #           },
    #           "perturbation_increment_config": {
    #               "MI-FGSM": {
    #                   "upper_bound": 0.02, "lower_bound": 0.008, "step": 0.001,
    #               }
    #           }
    #           }
    # batch_manager.init_batch(show_logo=True)
    # # global_recovery.start_recovery_mode("0qzx7UpZ")
    # security_evaluation = SecurityEvaluation(config)
    # security_evaluation.attack_perturbation_increment_test(use_img_file=True, use_raw_nparray_data=True)

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
