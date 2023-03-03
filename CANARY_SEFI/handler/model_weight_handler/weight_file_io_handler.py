import os
import torch
from CANARY_SEFI.task_manager import task_manager


def save_weight_to_temp(file_path, file_name, weight):
    full_path = task_manager.base_temp_path + "weight/" + file_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    full_path = full_path + file_name
    # Save model
    torch.save(weight, full_path)
    return
