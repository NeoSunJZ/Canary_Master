import numpy as np
from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from canary_sefi.core.function.helper.multi_db import use_multi_database
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager
from component_manager import init_component_manager

if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 10, "dataset": "ILSVRC-2012",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "I_FGSM": [
                "Alexnet(ImageNet)",
                "VGG(ImageNet)",
            ],
        },
        "model_list": ["Alexnet(ImageNet)", "VGG(ImageNet)"],
        "trans_list": {
            "I_FGSM": {
                "jpeg": ["Alexnet(ImageNet)", "VGG(ImageNet)"],
                "tvm": ["Alexnet(ImageNet)", "VGG(ImageNet)"],
                "quantize": ["Alexnet(ImageNet)", "VGG(ImageNet)"],
            },
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
            "I_FGSM": {
                "clip_min": 0,
                "clip_max": 1,
                "eps_iter": 2.5 * ((1 / 255) / 100),
                "nb_iter": 100,
                "norm": np.inf,
                "attack_type": "UNTARGETED",
                "epsilon": 1 / 255,
            }
        },
        "trans_config": {
            "jpeg": {"quality": 50},
            "tvm": {},
            "quantize": {}
        }
    }
    task_manager.init_task(show_logo=True, run_device="cuda")

    # 多数据库模式
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    security_evaluation = SecurityEvaluation(example_config)
    # security_evaluation.model_inference_capability_test_and_evaluation()
    # security_evaluation.adv_example_generate()
    # security_evaluation.attack_test_and_evaluation(use_raw_nparray_data=True)
    # security_evaluation.capability_evaluation(use_raw_nparray_data=True)

    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)

    # trans_full_test
    security_evaluation.trans_full_test(use_raw_nparray_data=True)

    # security_evaluation.adv_trans_generate()
    # security_evaluation.trans_test_and_evaluation(use_raw_nparray_data=True)
