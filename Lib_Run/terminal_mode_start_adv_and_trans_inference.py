import numpy as np

from CANARY_SEFI.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from CANARY_SEFI.core.function.helper.multi_db import use_multi_database
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager

if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 100, "dataset": "ILSVRC-2012",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "PNA_SIM": {
                "Alexnet(ImageNet)": ["jpeg", "tvm", "quantize"],
                "VGG(ImageNet)": ["jpeg", "tvm", "quantize"],
            },
        },
        "model_list": ["Alexnet(ImageNet)", "VGG(ImageNet)"],
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
                "epsilon": 1 / 255,
                "T": 100,
                "attack_type": "UNTARGETED",
                "tlabel": None
            }
        },
        "trans_config": {
            "jpeg": {

            },
            "tvm": {},
            "quantize": {}
        }
    }
    task_manager.init_task(task_token="5Jrs9nDZ", show_logo=True, run_device="cuda")

    # 多数据库模式
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    security_evaluation = SecurityEvaluation(example_config)
    # security_evaluation.adv_example_generate()
    # security_evaluation.model_inference_capability_test_and_evaluation()
    # security_evaluation.attack_test_and_evaluation(use_raw_nparray_data=True)
    # security_evaluation.capability_evaluation(use_raw_nparray_data=True)

    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)

    # trans_full_test
    security_evaluation.trans_full_test(use_raw_nparray_data=True)
