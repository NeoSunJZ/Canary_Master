from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.task_manager import task_manager
from utils import load_test_config
from component_manager import init_component_manager


if __name__ == "__main__":
    init_component_manager()
    config = load_test_config(attack_config="BA", data_config="ILSVRC2012-1000-SEED", model_config="IMAGENET-15",
                              attack_batch_config="BA-IMAGENET-15-RTX3090", model_batch_config="IMAGENET-15-RTX3090")
    task_manager.init_task(show_logo=True, run_device="cuda")
    global_recovery.start_recovery_mode(task_token="j3tCl0LF")
    security_evaluation = SecurityEvaluation(config)
    # security_evaluation.adv_example_generate()
    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)



