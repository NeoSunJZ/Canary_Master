import json
from datetime import datetime

from flask import Blueprint, Response

from CANARY_SEFI.copyright import get_system_version
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, SubComponentType, \
    AttackComponentAttributeType
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager

client_declaration = Blueprint('client_declaration', __name__)


@client_declaration.route('/getDeclaration', methods=['GET', 'POST'])
def get_declaration():
    config = config_manager.config
    declaration = {
        "version": get_system_version(),
        "appName": config.get("appName", 'Default Project Name'),
        "appDesc": config.get("appDesc", 'Default Project Description'),
        "registered_component": {
            "attacker_list": get_attacker_component_list(),
            "model_list": get_model_component_list(),
            "dataset_list": get_dataset_component_list()
        },
        "datatime": str(datetime.now())
    }
    return Response(json.dumps(declaration, ensure_ascii=False), mimetype='application/json')


def get_attacker_component_list():
    attacker_component_list = []
    for attack_method in SEFI_component_manager.attack_method_list:
        atk_component = SEFI_component_manager.attack_method_list[attack_method]
        attacker_component_list.append({
            "attacker_name": attack_method,
            "info": {
                "attack_type": atk_component.get(AttackComponentAttributeType.ATTACK_TYPE, None, True),
                "attack_config_params": atk_component.get(
                    ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS.value +
                    AttackComponentAttributeType.CONFIG_PARAMS.value, None, True),
                "is_class": atk_component.get(AttackComponentAttributeType.IS_INCLASS, None, True),
                "model_var_name": atk_component.get(AttackComponentAttributeType.MODEL_VAR_NAME, None, True),
                "perturbation_budget_var_name": atk_component.get(AttackComponentAttributeType.PERTURBATION_BUDGET_VAR_NAME, None, True),
            },
            "sub_component": {
                "attack_func": True if atk_component.get(SubComponentType.ATTACK_FUNC, None, True) else False,
                "attacker_class": True if atk_component.get(SubComponentType.ATTACK_CLASS, None, True) else False,
                "attacker_config_params_handler": True if atk_component.get(
                    ComponentConfigHandlerType.ATTACK_CONFIG_PARAMS.value + SubComponentType.CONFIG_PARAMS_HANDLER.value,
                    None, True) else False,
            }
        })
    return attacker_component_list


def get_model_component_list():
    model_component_list = []
    for model in SEFI_component_manager.model_list:
        model_component = SEFI_component_manager.model_list[model]
        model_component_list.append({
            "model_name": model,
            "info": {
                "model_config_params": model_component.get(
                    ComponentConfigHandlerType.MODEL_CONFIG_PARAMS.value +
                    AttackComponentAttributeType.CONFIG_PARAMS.value, None, True),
                "img_process_config_params": model_component.get(
                    ComponentConfigHandlerType.IMG_PROCESS_CONFIG_PARAMS.value +
                    AttackComponentAttributeType.CONFIG_PARAMS.value, None, True),
            },
            "sub_component": {
                "model_create_func": True if model_component.get(SubComponentType.MODEL_CREATE_FUNC, None, True) else False,
                "inference_detector": True if model_component.get(SubComponentType.MODEL_INFERENCE_DETECTOR, None, True) else False,
                "img_preprocessor": True if model_component.get(SubComponentType.IMG_PREPROCESSOR, None, True) else False,
                "img_reverse_processor": True if model_component.get(SubComponentType.IMG_REVERSE_PROCESSOR, None, True) else False,
                "result_postprocessor": True if model_component.get(SubComponentType.RESULT_POSTPROCESSOR, None, True) else False,
                "model_config_params_handler": True if model_component.get(
                    ComponentConfigHandlerType.MODEL_CONFIG_PARAMS.value + SubComponentType.CONFIG_PARAMS_HANDLER.value,
                    None, True) else False,
                "img_process_config_params_handler": True if model_component.get(
                    ComponentConfigHandlerType.IMG_PROCESS_CONFIG_PARAMS.value + SubComponentType.CONFIG_PARAMS_HANDLER.value,
                    None, True) else False,
            }
        })
    return model_component_list


def get_dataset_component_list():
    dataset_component_list = []
    for dataset in SEFI_component_manager.dataset_list:
        dataset_component = SEFI_component_manager.dataset_list[dataset]
        dataset_component_list.append({
            "dataset_name": dataset,
            "info": {
            },
            "sub_component": {
                "dataset_loader_handler": True if dataset_component.get(SubComponentType.DATASET_LOADER, None, True) else False,
            }
        })
    return dataset_component_list
