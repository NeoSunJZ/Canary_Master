from CANARY_SEFI.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from CANARY_SEFI.core.function.helper.multi_db import use_multi_database
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.handler.helper.correctness_check import CorrectnessCheck
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from utils import load_test_config
from component_manager import init_component_manager


if __name__ == "__main__":
    init_component_manager()

    config = {
        "dataset_size": 10,
        "dataset": "ILSVRC-2012",
        "dataset_seed": 40376958655838027,
        "model_name": "Alexnet(ImageNet)",
        "model_args": {},
        "img_proc_args": {},
        "atk_name": "EAD",
        "atk_args": {
            "attack_type": 'UNTARGETED',
            "tlabel": None,
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 16/255,
            "binary_search_steps": 9,
            "steps": 1000,
            "initial_stepsize": 0.01,
            "confidence": 0.0,
            "initial_const": 0.001,
            "regularization": 0.01,
            "decision_rule": 'EN',
            "abort_early": True
        },
        "run_device": "cuda",
        "adv_example_generate_batch_size": 1
    }

    correctness_check = CorrectnessCheck(config)
    correctness_check.attack_method_correctness_test(start_num=0, save_adv_example=False)
