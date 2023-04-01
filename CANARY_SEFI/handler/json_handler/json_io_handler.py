import json
import os


def save_info_to_json_file(info, base_temp_path, file_name):
    if not os.path.exists(base_temp_path + "data/"):
        os.makedirs(base_temp_path + "data/")

    full_path = base_temp_path + "data/" + file_name
    with open(full_path, 'w', encoding='utf-8') as json_file:
        json.dump(info, json_file, indent=4, ensure_ascii=False)


def get_info_from_json_file(base_temp_path, file_name):
    full_path = base_temp_path + "data/" + file_name
    if not os.path.exists(full_path):
        return None
    with open(full_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)
