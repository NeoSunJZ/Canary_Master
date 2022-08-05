from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType


def model_implemented_component_self_check(model_component):
    check_conclusion = []
    inference_function = "READY"
    used_by_white_box_attack = "READY"

    # 检查创建模型方法与模型的推断器是否被正确实现
    if model_component.get("model_create_func") is None:
        check_conclusion.append("[ERROR] Missing required component: Model Creator NOT FOUND")
        inference_function = "ERROR"
        used_by_white_box_attack = "ERROR"
    else:
        # 参数处理器检查
        if model_component.get("model_args_handler") is None:
            check_conclusion.append("[NOTICE] No Args Handler Defined: Model Args Handler NOT FOUND")

    if model_component.get("inference_detector") is None:
        check_conclusion.append("[ERROR] Missing required component: Inference Detector NOT FOUND")
        inference_function = "ERROR"

    # 检查图片处理器相关实现
    if model_component.get("img_preprocessor") is not None:
        if model_component.get("img_reverse_processor") is None:
            check_conclusion.append("[WARNING] Missing required component: Img Reverse Processor NOT FOUND!"
                                    "When the image preprocessor exists, The image inverse processor is the KEY to "
                                    "convert the adv example into a normal picture")
            used_by_white_box_attack = "WARNING"
        if model_component.get("result_postprocessor") is None:
            check_conclusion.append("[NOTICE] Missing optional component: Result Postprocessor NOT FOUND")
        # 参数处理器检查
        if model_component.get("img_processing_args_handler") is None:
            check_conclusion.append("[NOTICE] No Args Handler Defined: Img Processing Args Handler NOT FOUND")
    else:
        check_conclusion.append("[NOTICE] Missing optional component: Img Preprocessor NOT FOUND")

    return check_conclusion, {inference_function: inference_function,
                              used_by_white_box_attack: used_by_white_box_attack}


def attack_implemented_component_self_check(atk_component):
    check_conclusion = []
    attacker = "READY"

    if atk_component.get("attack_func") is None:
        check_conclusion.append("[ERROR] Missing required component: Attack Function NOT FOUND")
        attacker = "ERROR"

    if atk_component.get("attacker_class", {}).get("class") is None:
        if atk_component.get("is_inclass", True):
            check_conclusion.append(
                "[ERROR] Missing required component: is_inclass set to True, But Attack Class NOT FOUND")
            attacker = "ERROR"

    # 参数处理器检查
    if atk_component.get(ComponentConfigHandlerType.ATTACK_PARAMS.value + "_handler") is None:
        check_conclusion.append("[NOTICE] No Params Handler Defined: ATTACK Params Handler NOT FOUND, Will use Default")

    return check_conclusion, {attacker: attacker}
