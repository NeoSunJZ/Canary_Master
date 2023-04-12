from canary_sefi.core.component.component_enum import SubComponentType, ComponentConfigHandlerType
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_sefi.core.component.default_component.params_handler import build_dict_with_json_args
from canary_sefi.evaluator.monitor.attack_effect import model_query_statistics


def get_model(model_name, model_init_args, run_device, model_query_logger=None):
    # 根据modelName寻找Model是否已经注册
    model_component = SEFI_component_manager.model_list.get(model_name)

    # 构建Model
    create_model_func = model_component.get(SubComponentType.MODEL_CREATE_FUNC)
    model_args_dict = build_dict_with_json_args(model_component,
                                                ComponentConfigHandlerType.MODEL_CONFIG_PARAMS,
                                                model_init_args, run_device)
    model = create_model_func(**model_args_dict)
    if model_query_logger is not None:
        model.register_forward_hook(model_query_statistics(model_query_logger, "forward"))
        model.register_backward_hook(model_query_statistics(model_query_logger, "backward"))
    return model