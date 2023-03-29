from CANARY_SEFI.core.component.component_dict import ComponentDict, add_dict, ComponentDictType
from CANARY_SEFI.core.component.component_enum import ComponentType


class ComponentManager:
    def __init__(self):
        self.model_list = ComponentDict({},
                                        dict_type=ComponentDictType.ComponentDict,
                                        component_type=ComponentType.MODEL)
        self.attack_method_list = ComponentDict({},
                                                dict_type=ComponentDictType.ComponentDict,
                                                component_type=ComponentType.ATTACK)
        self.dataset_list = ComponentDict({},
                                          dict_type=ComponentDictType.ComponentDict,
                                          component_type=ComponentType.DATASET)

    def add_all(self, sefi_component_list):
        for sefi_component in sefi_component_list:
            self.add(sefi_component)

    def add(self, sefi_component):
        self.model_list = add_dict(self.model_list, sefi_component.models)
        self.attack_method_list = add_dict(self.attack_method_list, sefi_component.attack_methods)
        self.dataset_list = add_dict(self.dataset_list, sefi_component.datasets)

    def debug(self):
        print(self.model_list)
        print(self.attack_method_list)
        print(self.dataset_list)


SEFI_component_manager = ComponentManager()
