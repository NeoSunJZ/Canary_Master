import json
import os

from flask import Blueprint, Response

from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType
from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.handler.helper.self_check import model_implemented_component_self_check, \
    attack_implemented_component_self_check

client_declaration = Blueprint('client_declaration', __name__)


def get_model_list(config):
    model_list = []
    for model in SEFI_component_manager.model_list:
        model_component = SEFI_component_manager.model_list[model]
        self_check, support_function = model_implemented_component_self_check(model_component)
        model_temp = {
            # 模型名
            'modelName': model,
            # 是否自定义了模型参数处理器
            'hasModelArgsHandler': model_component.get("model_args_handler") is not None,
            # 模型参数声明
            'modelParamsDesc': model_component.get("model_args_handler_params"),

            # 是否自定义了图片预处理器、逆处理器（用于对抗样本生成时还原为原始图片）、结果后处理器
            'hasImgPreprocessor': model_component.get("img_preprocessor") is not None,
            'hasResultPostprocessor': model_component.get("result_postprocessor") is not None,
            'hasImgReverseProcessor': model_component.get("img_reverse_processor") is not None,

            # 是否自定义了图像处理参数处理器
            'hasImgProcessingArgsHandler': model_component.get("img_processing_args_handler") is not None,
            # 图像处理参数声明
            'imgProcessingParamsDesc': model_component.get("img_processing_args_handler_params"),

            # 预测器支持类型与返回类型
            'inferenceDetector': {
                "supportType": model_component.get("inference_detector", {}).get("support_type", None),
                "returnType": model_component.get("inference_detector", {}).get("return_type", None),
            },
            # 状态
            'self-check': self_check,
            'supportFunction': support_function,
            "supportApi": {
                "/inferenceDetector/inferenceImg": {
                    "isSupport": config.get("modelSupportApi", {}).get(model, {}).get("/inferenceDetector/img",
                                                                                      True) and support_function.get(
                        "inference_function") == "READY",
                    "paramsDesc": {"modelArgs": {"type": "ModelArgs"}, "targetArgs": {"type": "TargetArgs"},
                                   "modelName": {"type": "ModelName"}, "img": {"type": "File", "desc": "请上传需要识别的图片"}}
                },
                "/test/adversarialExamplesAttack/transferAttackTest": {
                    "isSupport": config.get("modelSupportApi", {}).get(model, {}).get(
                        "/test/adversarialExamplesAttack/transferAttackTest", True) and support_function.get(
                        "inference_function") == "READY",
                    "paramsDesc": {"modelArgs": {"type": "ModelArgs"}, "targetArgs": {"type": "TargetArgs"},
                                   "modelName": {"type": "ModelName"},
                                   "imgList": {"type": "ImgList", "desc": "请输入数据集对应的图片文件名列表，必须为数组"},
                                   "datasetName": {"type": "DatasetName", "desc": "请输入数据集名称"},
                                   "task_token": {"type": "Token", "desc": "请输入对抗样本批次号"}}
                }
            }
        }
        model_list.append(model_temp)
    return model_list


def get_attack_method_list(config):
    attack_method_list = []
    for attack_method in SEFI_component_manager.attack_method_list:
        atk_component = SEFI_component_manager.attack_method_list[attack_method]
        self_check, support_function = attack_implemented_component_self_check(atk_component)

        attack_method_temp = {
            # 攻击名
            'attackMethodName': attack_method,
            # 攻击类型
            'attack_type': atk_component.get("attack_type"),
            # 攻击支持模型（白名单）
            'attackSupportModel': atk_component.get("support_model", []),
            'attackNoModel': config.get("attackConfig", {}).get(attack_method, {}).get("no_model", False),

            # 是否自定义了攻击参数处理器
            'hasAttackMethodArgsHandler': atk_component.get(ComponentConfigHandlerType.ATTACK_PARAMS.value + "_handler") is not None,
            # 攻击参数声明
            'attackMethodArgsHandlerParamsDesc': atk_component.get(ComponentConfigHandlerType.ATTACK_PARAMS.value + "_handler_params"),

            # 攻击需预先初始化类（非方法级）
            'attackerClassNeedInit': atk_component.get("is_inclass", True),

            # 状态
            'self-check': self_check,
            'supportFunction': support_function,

            'supportApi': {
                "/advExample/attack/img": {
                    "isSupport": config.get("attackSupportApi", {}).get(attack_method, {}).get("/advExample/attack/img",
                                                                                               False)
                                 and support_function.get("attacker") == "READY",
                    "paramsDesc": {"modelName": {"type": "ModelName"},
                                   "modelArgs": {"type": "ModelArgs"}, "targetArgs": {"type": "TargetArgs"},
                                   "attackArgs": {"type": "AttackArgs"},
                                   "attackMethodName": {"type": "attackMethodName"},
                                   "img": {"type": "File", "desc": "请上传需要识别的图片"}}
                },
                "/advExample/attack/batch/img": {
                    "isSupport": config.get("attackSupportApi", {}).get(attack_method, {}).get(
                        "/advExample/attack/batch/img", False)
                                 and support_function.get("attacker") == "READY",
                    "paramsDesc": {
                        "modelName": {"type": "ModelName"},
                        "modelArgs": {"type": "ModelArgs"},
                        "targetArgs": {"type": "TargetArgs"},
                        "attackArgs": {"type": "AttackArgs"},
                        "attackMethodName": {"type": "AttackMethodName"},
                        "datasetName": {"type": "DatasetName", "desc": "请输入数据集名称"},
                        "imgList": {"type": "ImgList", "desc": "请输入图片列表"}
                    }
                }
            }
        }
        attack_method_list.append(attack_method_temp)
    return attack_method_list


def get_dataset_list(config):
    dataset_list = config.get("dataset", None)
    for key in dataset_list:
        index = 0
        for files in os.listdir(dataset_list.get(key).get("path")):
            index = index + 1
        dataset_list.get(key)['count'] = index
    return dataset_list


@client_declaration.route('/getDeclaration', methods=['GET', 'POST'])
def get_declaration():
    config = config_manager.config
    model_list = get_model_list(config)
    attack_method_list = get_attack_method_list(config)
    dataset_list = get_dataset_list(config)
    declaration = {
        "version": "v2.1.0",
        "appName": config.get("appName", 'Default Project Name'),
        "appDesc": config.get("appDesc", 'Default Project Description'),
        "model": {
            "hasModel": model_list is not None,
            "modelList": model_list,
        },
        "attack": {
            "hasAttack": attack_method_list is not None,
            "attackList": attack_method_list,
        },
        "dataset": {
            "hasDataset": dataset_list is not None,
            "datasetList": dataset_list,
        }
    }

    return Response(json.dumps(declaration, ensure_ascii=False), mimetype='application/json')
