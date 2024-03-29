import torch

from canary_sefi.core.component.component_dict import ComponentDict, ComponentDictType
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType, SubComponentType, \
    ModelComponentAttributeType, AttackComponentAttributeType, TransComponentAttributeType, \
    DefenseComponentAttributeType
from canary_sefi.core.component.component_exception import ParamsHandlerComponentTypeError, ComponentReturnTypeError, \
    UtilComponentTypeError, ComponentTypeError


class SEFIComponent:
    def __init__(self):
        self.models = ComponentDict({},
                                    dict_type=ComponentDictType.ComponentDict,
                                    component_type=ComponentType.MODEL)
        self.attack_methods = ComponentDict({},
                                            dict_type=ComponentDictType.ComponentDict,
                                            component_type=ComponentType.ATTACK)
        self.datasets = ComponentDict({},
                                      dict_type=ComponentDictType.ComponentDict,
                                      component_type=ComponentType.DATASET)
        self.defense_methods = ComponentDict({},
                                             dict_type=ComponentDictType.ComponentDict,
                                             component_type=ComponentType.DEFENSE)
        self.trans_methods = ComponentDict({},
                                           dict_type=ComponentDictType.ComponentDict,
                                           component_type=ComponentType.TRANS)

    def model(self, name, no_torch_model_check=False):
        target_model = self.get_models(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                model = decorated(*args, **kwargs)
                if not no_torch_model_check and not isinstance(model, torch.nn.Module):
                    raise ComponentReturnTypeError(
                        sub_component_name=SubComponentType.MODEL_CREATE_FUNC,
                        component_name=name, component_type=ComponentType.MODEL,
                        need_type=torch.nn.Module, get_type=type(model))
                return model

            target_model[SubComponentType.MODEL_CREATE_FUNC] = inner
            return inner
        return wrapper

    def config_params_handler(self, handler_target, name, handler_type, use_default_handler=False, params=None):
        handler_target = ComponentType(handler_target)
        handler_type = ComponentConfigHandlerType(handler_type)

        target = None
        if handler_target == ComponentType.MODEL:
            target = self.get_models(name)
            if handler_type not in (ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
                                    ComponentConfigHandlerType.IMG_PROCESS_CONFIG_PARAMS):
                raise ParamsHandlerComponentTypeError(component_name=name, component_type=handler_target, error_type=handler_type)
            target[handler_type.value + ModelComponentAttributeType.CONFIG_PARAMS.value] = params

        elif handler_target == ComponentType.ATTACK:
            target = self.get_attack_methods(name)
            if handler_type not in (ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS, ):
                raise ParamsHandlerComponentTypeError(component_name=name, component_type=handler_target, error_type=handler_type)
            target[handler_type.value + AttackComponentAttributeType.CONFIG_PARAMS.value] = params
        elif handler_target == ComponentType.DEFENSE:
            target = self.get_defense_methods(name)
            if handler_type not in (ComponentConfigHandlerType.DEFENSE_CONFIG_PARAMS,):
                raise ParamsHandlerComponentTypeError(component_name=name, component_type=handler_target, error_type=handler_type)
            target[handler_type.value + DefenseComponentAttributeType.CONFIG_PARAMS.value] = params
        elif handler_target == ComponentType.TRANS:
            target = self.get_trans_methods(name)
            if handler_type not in (ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,):
                raise ParamsHandlerComponentTypeError(component_name=name, component_type=handler_target, error_type=handler_type)
            target[handler_type.value + TransComponentAttributeType.CONFIG_PARAMS.value] = params

        else:
            raise ComponentTypeError(component_name=name, component_type=handler_target)

        def wrapper(decorated):
            if use_default_handler:
                return decorated

            def inner(*args, **kwargs):
                args_dict = decorated(*args, **kwargs)
                if not isinstance(args_dict, dict):
                    raise ComponentReturnTypeError(
                        sub_component_name=handler_type.value + SubComponentType.CONFIG_PARAMS_HANDLER.value,
                        component_name=name, component_type=handler_type,
                        need_type=type(dict), get_type=type(args_dict))
                return args_dict

            target[handler_type.value + SubComponentType.CONFIG_PARAMS_HANDLER.value] = inner
            return inner

        return wrapper

    def util(self, util_target, util_type, name):
        util_target = ComponentType(util_target)
        util_type = SubComponentType(util_type)

        targets = []
        if util_target == ComponentType.MODEL:
            if util_type not in (
                    SubComponentType.MODEL_TARGET_LAYERS_GETTER,
                    SubComponentType.MODEL_INFERENCE_DETECTOR,
                    SubComponentType.IMG_PREPROCESSOR,
                    SubComponentType.IMG_REVERSE_PROCESSOR,
                    SubComponentType.RESULT_POSTPROCESSOR):
                raise UtilComponentTypeError(component_name=name, component_type=util_target, error_type=util_type)
            if isinstance(name, list):
                for item in name:
                    targets.append(self.get_models(item))
            else:
                targets.append(self.get_models(name))

        elif util_target == ComponentType.ATTACK:
            if util_type not in ():
                raise UtilComponentTypeError(component_name=name, component_type=util_target, error_type=util_type)
            if isinstance(name, list):
                for item in name:
                    targets.append(self.get_models(item))
            else:
                targets.append(self.get_models(name))

        elif util_target == ComponentType.DATASET:
            if util_type not in (
                    SubComponentType.DATASET_LOADER, ):
                raise UtilComponentTypeError(component_name=name, component_type=util_target, error_type=util_type)
            if isinstance(name, list):
                for item in name:
                    targets.append(self.get_datasets(item))
            else:
                targets.append(self.get_datasets(name))

        else:
            raise ComponentTypeError(component_name=name, component_type=util_target)

        def wrapper(decorated):
            for target in targets:
                target[util_type] = decorated

            def inner(*args, **kwargs):
                return decorated(*args, **kwargs)
            return inner
        return wrapper

    def attack(self, name, is_inclass=False, support_model=None, attack_type=None,
               model_var_name="model",
               model_require=True,
               perturbation_budget_var_name=None):
        target_attack_method = self.get_attack_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                attack_method = decorated(*args, **kwargs)
                return attack_method

            target_attack_method[SubComponentType.ATTACK_FUNC] = inner
            target_attack_method[AttackComponentAttributeType.SUPPORT_MODEL] = support_model
            target_attack_method[AttackComponentAttributeType.IS_INCLASS] = is_inclass
            target_attack_method[AttackComponentAttributeType.ATTACK_TYPE] = attack_type

            target_attack_method[AttackComponentAttributeType.MODEL_VAR_NAME] = model_var_name
            target_attack_method[AttackComponentAttributeType.MODEL_REQUIRE] = model_require

            target_attack_method[AttackComponentAttributeType.SUPPORT_MODEL] = support_model
            # 扰动预算变量名
            target_attack_method[
                AttackComponentAttributeType.PERTURBATION_BUDGET_VAR_NAME] = perturbation_budget_var_name
            return inner

        return wrapper

    def defense(self, name, is_inclass, support_model=[], defense_type=''):
        target_defense_method = self.get_defense_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                defense_method = decorated(*args, **kwargs)
                return defense_method

            target_defense_method[SubComponentType.DEFENSE_FUNC] = inner
            target_defense_method[DefenseComponentAttributeType.SUPPORT_MODEL] = support_model
            target_defense_method[DefenseComponentAttributeType.IS_INCLASS] = is_inclass
            target_defense_method[DefenseComponentAttributeType.DEFENSE_TYPE] = defense_type

            return inner

        return wrapper

    def trans(self, name, is_inclass, trans_type=''):
        target_trans_method = self.get_trans_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                trans_method = decorated(*args, **kwargs)
                return trans_method

            target_trans_method[SubComponentType.TRANS_FUNC] = inner
            target_trans_method[TransComponentAttributeType.IS_INCLASS] = is_inclass
            target_trans_method[TransComponentAttributeType.TRANS_TYPE] = trans_type

            return inner

        return wrapper

    def attack_init(self, name):
        target_attack_method = self.get_attack_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                attack_init = decorated(*args, **kwargs)
                return attack_init

            target_attack_method[SubComponentType.ATTACK_INIT] = inner
            return inner

        return wrapper

    def defense_init(self, name):
        target_defense_method = self.get_defense_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                defense_init = decorated(*args, **kwargs)
                return defense_init

            target_defense_method[SubComponentType.DEFENSE_INIT] = inner
            return inner

        return wrapper

    def trans_init(self, name):
        target_trans_method = self.get_trans_methods(name)

        def wrapper(decorated):
            def inner(*args, **kwargs):
                trans_init = decorated(*args, **kwargs)
                return trans_init

            target_trans_method[SubComponentType.TRANS_INIT] = inner
            return inner

        return wrapper

    def attacker_class(self, attack_name,
                       model_var_name="model",
                       perturbation_budget_var_name=None):
        target_attack_method = self.get_attack_methods(attack_name)

        def wrapper(decorated):
            target_attack_method[SubComponentType.ATTACK_CLASS] = decorated
            target_attack_method[AttackComponentAttributeType.PERTURBATION_BUDGET_VAR_NAME] = perturbation_budget_var_name
            target_attack_method[AttackComponentAttributeType.MODEL_VAR_NAME] = model_var_name
            return decorated

        return wrapper

    def defense_class(self, defense_name):
        target_defense_method = self.defense_methods.get(defense_name, default=None, allow_not_exist=True)
        if target_defense_method is None:
            self.defense_methods[defense_name] = {}
            target_defense_method = self.defense_methods.get(defense_name)

        def wrapper(decorated):
            target_defense_method[SubComponentType.DEFENSE_CLASS] = decorated
            return decorated

        return wrapper

    def trans_class(self, trans_name):
        target_trans_method = self.trans_methods.get(trans_name, default=None, allow_not_exist=True)
        if target_trans_method is None:
            self.trans_methods[trans_name] = {}
            target_trans_method = self.trans_methods.get(trans_name)

        def wrapper(decorated):
            target_trans_method[SubComponentType.TRANS_CLASS] = decorated
            return decorated

        return wrapper

    def get_models(self, name):
        target_model = self.models.get(name, default=None, allow_not_exist=True)
        if target_model is None:
            self.models[name] = ComponentDict({},
                                              dict_type=ComponentDictType.SubComponentDict,
                                              component_type=ComponentType.MODEL, component_name=name)
            target_model = self.models.get(name)
        return target_model

    def get_attack_methods(self, name):
        target_attack_method = self.attack_methods.get(name, default=None, allow_not_exist=True)
        if target_attack_method is None:
            self.attack_methods[name] = ComponentDict({},
                                                      dict_type=ComponentDictType.SubComponentDict,
                                                      component_type=ComponentType.ATTACK, component_name=name)
            target_attack_method = self.attack_methods.get(name)
        return target_attack_method

    def get_defense_methods(self, name):
        target_defense_method = self.defense_methods.get(name, default=None, allow_not_exist=True)
        if target_defense_method is None:
            self.defense_methods[name] = ComponentDict({},
                                                       dict_type=ComponentDictType.SubComponentDict,
                                                       component_type=ComponentType.DEFENSE, component_name=name)
            target_defense_method = self.defense_methods.get(name)
        return target_defense_method

    def get_trans_methods(self, name):
        target_trans_method = self.trans_methods.get(name, default=None, allow_not_exist=True)
        if target_trans_method is None:
            self.trans_methods[name] = ComponentDict({},
                                                     dict_type=ComponentDictType.SubComponentDict,
                                                     component_type=ComponentType.TRANS, component_name=name)
            target_trans_method = self.trans_methods.get(name)
        return target_trans_method

    def get_datasets(self, name):
        target_datasets = self.datasets.get(name, default=None, allow_not_exist=True)
        if target_datasets is None:
            self.datasets[name] = ComponentDict({},
                                                dict_type=ComponentDictType.SubComponentDict,
                                                component_type=ComponentType.DATASET, component_name=name)
            target_datasets = self.datasets.get(name)
        return target_datasets
