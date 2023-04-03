from CANARY_SEFI.core.function.training_and_evaluation import adv_defense_training
from CANARY_SEFI.task_manager import task_manager
from CANARY_SEFI.core.function.init_dataset import init_dataset
from CANARY_SEFI.handler.json_handler.json_io_handler import save_info_to_json_file, get_info_from_json_file


class DefenseEvaluation:

    def __init__(self, config=None):
        if config is None:
            config = get_info_from_json_file(task_manager.base_temp_path, "config.json")
        else:
            save_info_to_json_file(config, task_manager.base_temp_path, "config.json")
        self.dataset_info = init_dataset(config.get("dataset"), config.get("dataset_size"),
                                         config.get("dataset_seed", None),config.get("is_train", True))

        self.model_list = config.get("model_list", None)
        self.defense_list = config.get("defense_list", None)

        self.model_config = config.get("model_config", None)
        self.defense_config = config.get("defense_config", None)
        self.img_proc_config = config.get("img_proc_config", None)

    def adv_defense_training(self):
        # 对抗防御训练
        adv_defense_training(self.dataset_info, self.defense_list, self.defense_config, self.model_config,
                             self.img_proc_config)
        task_manager.test_data_logger.finish()
