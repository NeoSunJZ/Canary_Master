import json
import os


def load_attack_list():
    current_path = os.path.abspath(__file__)
    target_directory = current_path[
                       :current_path.index('get_attack_file.py')] + "file_path.json"
    with open(target_directory, 'r') as file:
        attack_list = json.load(file)
    return attack_list


def get_attack_path(attack_method):
    attack_list = load_attack_list()
    if attack_method in attack_list:
        return attack_list[attack_method]
    else:
        return None
