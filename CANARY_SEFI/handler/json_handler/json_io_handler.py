import json
import os

from CANARY_SEFI.task_manager import task_manager


def save_info_to_json_file(info, file_name):
    if not os.path.exists(task_manager.base_temp_path + "data/"):
        os.makedirs(task_manager.base_temp_path + "data/")

    full_path = task_manager.base_temp_path + "data/" + file_name
    with open(full_path, 'w', encoding='utf-8') as json_file:
        json.dump(info, json_file, indent=4, ensure_ascii=False)


def get_info_from_json_file(file_name):
    full_path = task_manager.base_temp_path + "data/" + file_name
    with open(full_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)