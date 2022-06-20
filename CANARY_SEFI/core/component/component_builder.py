import json

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager


def get_model(model_name, model_init_args):
    # 根据modelName寻找Model是否已经注册
    model_component = SEFI_component_manager.model_list.get(model_name)
    if model_component is None:
        # 未找到指定的Model
        return None

    # 构建Model
    create_model_func = model_component.get("model_create_func")
    model_args_dict = build_dict_with_json_args(model_component, "model", model_init_args)
    model = create_model_func(**model_args_dict)
    return model


def build_dict_with_json_args(component, component_name, args):
    # 当传入值已经是dict时，无需进行任何处理
    if type(args) == dict:
        return args
    args_handler_func = component.get(component_name + "_args_handler")
    target_args_dict = {}
    # 当json参数存在时构造dict
    if args is not None:
        # 参数转换构造器存在就用自己的，否则用默认的
        if args_handler_func is not None:
            target_args_dict = args_handler_func(args)
        else:
            try:
                target_args_dict = json.loads(args)
            except Exception as e:
                raise TypeError("The default parameter converter is used but the input is not JSON")
    return target_args_dict

