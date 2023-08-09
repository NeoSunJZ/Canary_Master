import os
import torch

from canary_sefi.evaluator.logger.adv_training_weight_path_info_handler import add_adv_training_weight_file_path_log
from canary_sefi.task_manager import task_manager


def save_weight_to_temp(model_name, defense_name, epoch_cursor, file_path, file_name, weight):
    full_path = task_manager.base_temp_path + "weight/" + file_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    full_path = full_path + file_name
    # Save model
    torch.save(weight, full_path)
    add_adv_training_weight_file_path_log(model_name, defense_name, epoch_cursor, full_path)
    return
