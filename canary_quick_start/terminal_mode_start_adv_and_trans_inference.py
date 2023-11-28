import numpy as np
from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from .core.function.helper.multi_db import use_multi_database
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_lib import canary_lib  # Canary Lib

SEFI_component_manager.add_all(canary_lib)

if __name__ == "__main__":
    example_config = {
        "dataset_size": 1000, "dataset": "CIFAR10",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "I_FGSM": [
                "ResNet(CIFAR-10)_NATURAL",
                "ResNet(CIFAR-10)_TRADES",
                "ResNet(CIFAR-10)_MART",
                "ResNet(CIFAR-10)_NAT",
            ],
        },
        "model_list": [
            "ResNet(CIFAR-10)_NATURAL",
            "ResNet(CIFAR-10)_TRADES",
            "ResNet(CIFAR-10)_MART",
            "ResNet(CIFAR-10)_NAT",
        ],
        "defense_model_list": {
            "ResNet(CIFAR-10)": [
                "TRADES", "MART", "NAT"
            ]
        },
        "img_proc_config": {
        },
        "trans_list": {
            "I_FGSM": {
                "jpeg": [
                    "ResNet(CIFAR-10)_NATURAL",
                    "ResNet(CIFAR-10)_TRADES",
                    "ResNet(CIFAR-10)_MART",
                    "ResNet(CIFAR-10)_NAT",
                ],
                "tvm": [
                    "ResNet(CIFAR-10)_NATURAL",
                    "ResNet(CIFAR-10)_TRADES",
                    "ResNet(CIFAR-10)_MART",
                    "ResNet(CIFAR-10)_NAT",
                ],
                "quantize": [
                    "ResNet(CIFAR-10)_NATURAL",
                    "ResNet(CIFAR-10)_TRADES",
                    "ResNet(CIFAR-10)_MART",
                    "ResNet(CIFAR-10)_NAT",
                ],
                "quilting": [
                    "ResNet(CIFAR-10)_NATURAL",
                    "ResNet(CIFAR-10)_TRADES",
                    "ResNet(CIFAR-10)_MART",
                    "ResNet(CIFAR-10)_NAT",
                ],
            },
        },
        "model_config": {
            "ResNet(CIFAR-10)_MART": {
                "pretrained_file": "/home/zhangda/cifar-10_model/AT_MART_ResNet(CIFAR-10)_final.pt"
            },
            "ResNet(CIFAR-10)_NAT": {
                "pretrained_file": "/home/zhangda/cifar-10_model/AT_NAT_ResNet(CIFAR-10)_final.pt"
            },
            "ResNet(CIFAR-10)_NATURAL": {
                "pretrained_file": "/home/zhangda/cifar-10_model/AT_NATURAL_ResNet(CIFAR-10)_final.pt"
            },
            "ResNet(CIFAR-10)_TRADES": {
                "pretrained_file": "/home/zhangda/cifar-10_model/AT_TRADES_ResNet(CIFAR-10)_final.pt"
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
            },
        },
        "trans_config": {
            "jpeg": {"quality": 50},
            "tvm": {},
            "quantize": {},
            "quilting": {"quilting_size": 3}
        }
    }
    task_manager.init_task(show_logo=True, run_device="cuda")

    # 多数据库模式
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    security_evaluation = SecurityEvaluation(example_config)

    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)
    # security_evaluation.model_inference_capability_test_and_evaluation()
    # security_evaluation.adv_example_generate()
    # security_evaluation.attack_test_and_evaluation(use_raw_nparray_data=True)
    # security_evaluation.capability_evaluation(use_raw_nparray_data=True)

    security_evaluation.trans_full_test(use_raw_nparray_data=True)
    # security_evaluation.adv_trans_generate()
    # security_evaluation.trans_test_and_evaluation(use_raw_nparray_data=True)
