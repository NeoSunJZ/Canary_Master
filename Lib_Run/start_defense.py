from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.function.basic.train_function import adv_defense_4_img_batch
from CANARY_SEFI.entity.dataset_info_entity import DatasetInfo
from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager
from Defense_Method.Adversarial_Training.trades import sefi_component as trades_component

if __name__ == "__main__":
    init_component_manager()
    SEFI_component_manager.add(trades_component)

    adv_defense_4_img_batch(defense_name="trades", defense_args={}, model_name="", model_args={}, img_proc_args={},
                            dataset_info=DatasetInfo(dataset_name="CIFAR-10", dataset_seed=40376958655838027))
