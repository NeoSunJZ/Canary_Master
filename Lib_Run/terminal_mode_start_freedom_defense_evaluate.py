from CANARY_SEFI.service.security_evaluation import SecurityEvaluation
from CANARY_SEFI.task_manager import task_manager
from component_manager import init_component_manager

if __name__ == "__main__":
    init_component_manager()
    example_config = {
        "dataset_size": 10, "dataset": "CIFAR-10",
        "dataset_seed": 40376958655838027,
        "attacker_list": {
            "FGM": [
                "DenseNet(CIFAR-10)",
            ],
        },
        "defense_model_list": {
            "DenseNet(CIFAR-10)": [
                "TRADES",
            ]
        },
        "model_list": [
            "DenseNet(CIFAR-10)",
        ],
        "model_config": {
            "DenseNet(CIFAR-10)_TRADES": {
                "pretrained_file": "E:\github_clone_workspace\Canary_Master\Model\CIFAR10\DenseNet\weight\densenet161.pt"
            }
        },
        "img_proc_config": {
        },
        "attacker_config": {
            "FGM": {
                "clip_min": 0,
                "clip_max": 1,
                "epsilon": 16 / 255,
                "attack_type": 'UNTARGETED',
            }
        },
        "inference_batch_config": {
            "DenseNet(CIFAR-10)": 1,
        },
        "adv_example_generate_batch_config": {
            "FGM": {
                "DenseNet(CIFAR-10)": 1,
            },
        },

    }
    task_manager.init_task(show_logo=True,  task_token='aTsu61po', run_device="cuda")
    security_evaluation = SecurityEvaluation(example_config)
    security_evaluation.defense_test_and_evaluation(use_raw_nparray_data=True)