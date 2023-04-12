from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.function.basic.train_function import adv_defense_4_img_batch
from canary_sefi.entity.dataset_info_entity import DatasetInfo
from component_manager import init_component_manager
from Defense_Method.Adversarial_Training.trades import sefi_component as trades_component

if __name__ == "__main__":
    init_component_manager()
    SEFI_component_manager.add(trades_component)

    adv_defense_4_img_batch(defense_name="trades", defense_args={}, model_name="DenseNet(CIFAR-10)",
                            model_args={"is_pretrained":False, "no_normalize_layer":True}, img_proc_args={},
                            dataset_info=DatasetInfo(dataset_name="CIFAR-10", dataset_seed=40376958655838027, dataset_size=100),
                            run_device="cuda")
