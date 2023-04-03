# from add_path import append_sys_path
# append_sys_path("/home/xiongwenqi/Canary/")

from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager
from CANARY_SEFI.service.defense_evaluation import DefenseEvaluation


if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 10, "dataset": "CIFAR-10",
        "dataset_seed": 40376958655838027,
        "is_train": True,
        "defense_list": {
            "natural":[
                "ResNet(CIFAR-10)"
            ],
            "trades": [
                "ResNet(CIFAR-10)"
            ],
        },
        "defense_config": {
            "natural": {
                "epochs": 5,
                "log_interval": 5
            },
            "trades": {
                "epochs": 5,
                "log_interval": 5
            },
        },
        "model_config": {
            "ResNet(CIFAR-10)": {
                "is_pretrained": False,
                "no_normalize_layer": True
            },
        },
        "img_proc_config": {
            "ResNet(CIFAR-10)": {
                "batch_size": 2
            },
        },
    }
    task_manager.init_task(show_logo=True, run_device="cuda")
    security_evaluation = DefenseEvaluation(example_config)
    security_evaluation.adv_defense_training()

# "pretrained_file": "/home/xiongwenqi/temp/IaAk7NB8/weight/DenseNet(CIFAR-10)/AT_DenseNet("
# "CIFAR-10)_CIFAR-10_IaAk7NB8.pt",
