from copy import deepcopy


def add_dict(dict_1, dict_2):
    dic = deepcopy(dict_1)
    for key in dict_2.keys():
        if key in dic:
            dic[key].update(dict_2[key])
        else:
            dic[key] = dict_2[key]
    return dic


class ComponentManager:
    def __init__(self):
        self.model_list = {}
        self.attack_method_list = {}
        self.dataset_list = {}

    def add(self, sefi_component):
        self.model_list = add_dict(self.model_list, sefi_component.models)
        self.attack_method_list = add_dict(self.attack_method_list, sefi_component.attack_methods)
        self.dataset_list = add_dict(self.dataset_list, sefi_component.datasets)

    def debug(self):
        print(self.model_list)
        print(self.attack_method_list)
        print(self.dataset_list)

SEFI_component_manager = ComponentManager()
