from CANARY_SEFI.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from CANARY_SEFI.core.function.helper.multi_db import use_multi_database
from CANARY_SEFI.core.function.helper.recovery import global_recovery
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from utils import load_test_config
from component_manager import init_component_manager


if __name__ == "__main__":
    init_component_manager()
    config = load_test_config(attack_config="IFGSM", data_config="ILSVRC2012-1000-SEED", model_config="IMAGENET-15",
                              attack_batch_config="IFGSM-IMAGENET-15-RTX3090", model_batch_config="IMAGENET-15-RTX3090")
    # 初始化，以下二者必选其一
    #  task_manager.init_task(show_logo=True, run_device="cuda")  # 正常的初始化
    global_recovery.start_recovery_mode(task_token="moqX3ie6", show_logo=True, run_device="cuda")  # 恢复任务时的初始化

    # 多数据库模式
    use_multi_database(mode=MultiDatabaseMode.EACH_ATTACK_ISOLATE_DB, center_database_token="Bms4PuYC",
                       multi_database_config={})

    security_evaluation = SecurityEvaluation(config)
    # security_evaluation.adv_example_generate()
    # security_evaluation.model_inference_capability_test_and_evaluation()
    security_evaluation.attack_full_test(use_img_file=False, use_raw_nparray_data=True)
