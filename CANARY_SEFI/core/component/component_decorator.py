from enum import Enum

import torch

from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType


class SEFIComponent:
    def __init__(self):
        self.models = {}
        self.attack_methods = {}
        self.datasets = {}
        self.lock = False

    def model(self, name, no_torch_model_check=False):
        target_model = self.get_models(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                model = decorated(*args, **kwargs)
                if not no_torch_model_check and not isinstance(model, torch.nn.Module):
                    raise TypeError("[ Config Error ] The return model type must be a torch.nn.Module , but it was not")
                return model

            target_model['model_create_func'] = inner
            return inner

        return wrapper

    def inference_detector(self, model_name, support_type=None, return_type=None):
        target_model = self.get_models(model_name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                result = decorated(*args, **kwargs)
                return result

            target_model["inference_detector"] = {
                "func": inner,
                "support_type": support_type,
                "return_type": return_type,
            }
            return inner

        return wrapper

    def config_params_handler(self, handler_target, name, args_type, use_default_handler=False, params=None):
        target = None
        if handler_target == ComponentType.MODEL:
            target = self.get_models(name)
            if args_type not in ('model_args', 'target_args'):
                raise Exception("[ Config Error ] Illegal Model Args Handler Type")
        elif handler_target == ComponentType.ATTACK:
            target = self.get_attack_methods(name)
            if args_type not in (ComponentConfigHandlerType.ATTACK_PARAMS, 'target_args'):
                raise Exception("[ Config Error ] Illegal Attack Args Handler Type")

        target[args_type.value + "_handler_params"] = params

        def wrapper(decorated):
            if use_default_handler:
                return decorated

            def inner(*args, **kwargs):
                args_dict = decorated(*args, **kwargs)
                if not isinstance(args_dict, dict):
                    raise TypeError("[ Config Error ] The config params handler return type must be a dict , but it was not")
                return args_dict

            target[args_type + '_handler'] = inner
            return inner

        return wrapper

    def util(self, util_target, util_type, name):
        target = None
        if util_target == "model":
            target = self.get_models(name)
        elif util_target == "attack":
            target = self.get_attack_methods(name)
        elif util_target == "dataset":
            target = self.get_datasets(name)


        def wrapper(decorated):
            target[util_type] = decorated

            def inner(*args):
                return decorated(*args)

            return inner

        return wrapper

    def attack(self, name, is_inclass, support_model=[], attack_type='', perturbation_budget_var_name=None):
        target_attack_method = self.get_attack_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                attack_method = decorated(*args, **kwargs)
                return attack_method

            target_attack_method['attack_func'] = inner
            target_attack_method['support_model'] = support_model
            target_attack_method['is_inclass'] = is_inclass
            target_attack_method['attack_type'] = attack_type

            # 扰动预算变量名
            target_attack_method['perturbation_budget_var_name'] = perturbation_budget_var_name

            return inner

        return wrapper

    def defense(self, name, is_inclass, support_model=[], defense_type=''):
        target_defense_method = self.get_defense_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                defense_method = decorated(*args, **kwargs)
                return defense_method

            target_defense_method['attack_func'] = inner
            target_defense_method['support_model'] = support_model
            target_defense_method['is_inclass'] = is_inclass
            target_defense_method['attack_type'] = defense_type

            return inner

        return wrapper

    def attack_init(self, name):
        target_attack_method = self.get_attack_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                attack_init = decorated(*args, **kwargs)
                return attack_init

            target_attack_method['attack_init'] = inner
            return inner

        return wrapper

    def attacker_class(self, attack_name, attacker_class_model_var_name="model", perturbation_budget_var_name=None):
        target_attack_method = self.attack_methods.get(attack_name)
        if target_attack_method is None:
            self.attack_methods[attack_name] = {}
            target_attack_method = self.attack_methods.get(attack_name)

        def wrapper(decorated):
            target_attack_method['attacker_class'] = {
                "class": decorated,
                "attacker_class_model_var_name": attacker_class_model_var_name,
                "perturbation_budget_var_name": perturbation_budget_var_name,  # 扰动预算变量名
            }
            return decorated

        return wrapper

    def defense_class(self, defense_name):
        target_defense_method = self.defense_methods.get(defense_name)
        if target_defense_method is None:
            self.defense_methods[defense_name] = {}
            target_defense_method = self.defense_methods.get(defense_name)

        def wrapper(decorated):
            target_defense_method['defense_class'] = {
                "class": decorated,
            }
            return decorated

        return wrapper

    def get_models(self, name):
        target_model = self.models.get(name)
        if target_model is None:
            self.models[name] = {}
            target_model = self.models.get(name)
        return target_model

    def get_attack_methods(self, name):
        target_attack_method = self.attack_methods.get(name)
        if target_attack_method is None:
            self.attack_methods[name] = {}
            target_attack_method = self.attack_methods.get(name)
        return target_attack_method

    def get_defense_methods(self, name):
        target_defense_method = self.defense_methods.get(name)
        if target_defense_method is None:
            self.defense_methods[name] = {}
            target_defense_method = self.defense_methods.get(name)
        return target_defense_method

    def get_datasets(self, name):
        target_datasets = self.datasets.get(name)
        if target_datasets is None:
            self.datasets[name] = {}
            target_datasets = self.datasets.get(name)
        return target_datasets

    def attack_param(self):
        def wrapper(decorated):
            return decorated
        return wrapper
