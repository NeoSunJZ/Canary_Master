from CANARY_SEFI.core.function.basic.img_trans_function import adv_trans_4_img_batch
from CANARY_SEFI.evaluator.logger.attack_info_handler import find_attack_log_by_name_and_base_model
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager

task_manager.init_task(task_token='Vda1qvp5', show_logo=True, run_device="cuda")
atk_name = "PNA_SIM"
base_model = "EfficientNetV2(ImageNet)"
attack_log = find_attack_log_by_name_and_base_model(atk_name, base_model)
print(attack_log)

if __name__ == "__main__":
    init_component_manager()
    trans_args = {'args': {}}
    adv_trans_4_img_batch(trans_name="quantize", trans_args={}, atk_log=attack_log)
