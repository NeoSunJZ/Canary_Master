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
        self.defense_method_list = {}

    def add_all(self, sefi_component_list):
        for sefi_component in sefi_component_list:
            self.add(sefi_component)

    def add(self, sefi_component):
        self.model_list = add_dict(self.model_list, sefi_component.models)
        self.attack_method_list = add_dict(self.attack_method_list, sefi_component.attack_methods)
        self.dataset_list = add_dict(self.dataset_list, sefi_component.datasets)
        self.defense_method_list = add_dict(self.defense_method_list, sefi_component.defense_methods)

    def debug(self):
        print(self.model_list)
        print(self.attack_method_list)
        print(self.dataset_list)
        print(self.defense_method_list)


SEFI_component_manager = ComponentManager()
