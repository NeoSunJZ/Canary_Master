import json
from CANARY_SEFI.core.component.component_enum import SubComponentType


def build_dict_with_json_args(component, handler_type, params, run_device=None):
    # 当传入值已经是dict时，无需进行任何处理
    if type(params) == dict:
        return add_run_device(params, run_device)
    params_handler_func = component.get(handler_type.value + SubComponentType.CONFIG_PARAMS_HANDLER.value)
    target_args_dict = None
    # 当json参数存在时构造dict
    if params is not None:
        # 参数转换构造器存在就用自己的，否则用默认的
        if params_handler_func is not None:
            target_args_dict = params_handler_func(params)
        else:
            try:
                target_args_dict = json.loads(params)
            except Exception as e:
                print(e)
                raise ValueError("[SEFI] User has NOT defined extra parameter handler "
                                 "AND default parameter converter is used BUT THE INPUT IS NOT VALID JSON")
    return add_run_device(target_args_dict, run_device)


def add_run_device(args, run_device=None):
    # 添加执行设备
    if run_device is not None:
        args['run_device'] = run_device
    return args

