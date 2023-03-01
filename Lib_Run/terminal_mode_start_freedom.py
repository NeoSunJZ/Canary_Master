from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager

if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 10, "dataset": "ILSVRC-2012",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "PNA_SIM": [
                "Alexnet(ImageNet)",  # 2012 Alex & Hinton
                "VGG(ImageNet)",  # 2014 牛津大学计算机视觉组
                "GoogLeNet(ImageNet)",  # 2014 谷歌
                "InceptionV3(ImageNet)",  # 2014:V1 2015:V3
                "ResNet(ImageNet)",  # 2015 脸书 何凯明
                "DenseNet(ImageNet)",  # 2016
                "SqueezeNet(ImageNet)",  # 2016
                "MobileNetV3(ImageNet)",  # 2017:V1 2019:V3 谷歌
                "ShuffleNetV2(ImageNet)",  # 2018:V2 旷视
                "MNASNet(ImageNet)",  # 2018
                "EfficientNetV2(ImageNet)",  # 2019
                "ViT(ImageNet)",  # 2020 谷歌
                "RegNet(ImageNet)",  # 2020 脸书 何凯明
                "SwinTransformer(ImageNet)",  # 2021
                "ConvNext(ImageNet)",  # 2022 脸书
            ],
        },
        "img_proc_config": {
            "EfficientNetV2(ImageNet)": {
                "img_size_h": 384,
                "img_size_w": 384
            },
            "InceptionV3(ImageNet)": {
                "img_size_h": 299,
                "img_size_w": 299
            },
        },
        "attacker_config": {
            "PNA_SIM": {
                "clip_min": 0,
                "clip_max": 1,
                "epsilon": 16 / 255,
                "attack_type": 'UNTARGETED',
            }
        },
    }
    task_manager.init_task(show_logo=True, run_device="cuda")
    security_evaluation = SecurityEvaluation(example_config)
    security_evaluation.adv_example_generate()
