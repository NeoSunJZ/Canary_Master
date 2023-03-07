from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager
from CANARY_SEFI.service.defense_evaluation import DefenseEvaluation

if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 10, "dataset": "CIFAR-10",
        "dataset_seed": 40376958655838027,
        "defense_list": {
            "trades": [
                "DenseNet(CIFAR-10)"
            ],
        },
        "defense_config": {
            "trades": {
                "epochs": 5
            },
        },
        "model_config": {
            "DenseNet(CIFAR-10)": {
                "is_pretrained": False,
                "no_normalize_layer": True
            },
        },
        "img_proc_config": {
        },
    }
    task_manager.init_task(show_logo=True, run_device="cuda")
    security_evaluation = DefenseEvaluation(example_config)
    security_evaluation.adv_defense_training()
