from enum import Enum


class ComponentType(Enum):
    ATTACK = "attack"
    MODEL = "model"
    DATASET = "dataset"
    DEFENSE = "defense"
    TRANS = "trans"


class SubComponentType(Enum):
    MODEL_CREATE_FUNC = "model_create_func"
    MODEL_INFERENCE_DETECTOR = "inference_detector"
    MODEL_TARGET_LAYERS_GETTER = "target_layers_getter"

    ATTACK_FUNC = "attack_func"
    ATTACK_CLASS = "attacker_class"
    ATTACK_INIT = "attack_init"

    TRANS_FUNC = "trans_func"
    TRANS_CLASS = "trans_class"
    TRANS_INIT = "trans_init"

    DEFENSE_FUNC = "defense_func"
    DEFENSE_CLASS = "defense_class"
    DEFENSE_INIT = "defense_init"

    CONFIG_PARAMS_HANDLER = "_config_params_handler"

    IMG_PREPROCESSOR = "img_preprocessor"
    IMG_REVERSE_PROCESSOR = "img_reverse_processor"
    RESULT_POSTPROCESSOR = "result_postprocessor"

    DATASET_LOADER = "dataset_loader_handler"


class ModelComponentAttributeType(Enum):
    CONFIG_PARAMS = "_config_params"


class AttackComponentAttributeType(Enum):
    CONFIG_PARAMS = "_config_params"
    SUPPORT_MODEL = "support_model"
    IS_INCLASS = "is_inclass"
    ATTACK_TYPE = "attack_type"
    MODEL_VAR_NAME = "model_var_name"
    MODEL_REQUIRE = "model_require"
    PERTURBATION_BUDGET_VAR_NAME = "perturbation_budget_var_name"


class DefenseComponentAttributeType(Enum):
    CONFIG_PARAMS = "config_params"
    SUPPORT_MODEL = "support_model"
    IS_INCLASS = "is_inclass"
    DEFENSE_TYPE = "defense_type"

class TransComponentAttributeType(Enum):
    CONFIG_PARAMS = "config_params"
    IS_INCLASS = "is_inclass"
    TRANS_TYPE = "trans_type"


class ComponentConfigHandlerType(Enum):
    ATTACK_CONFIG_PARAMS = "attack"
    MODEL_CONFIG_PARAMS = "model"
    IMG_PROCESS_CONFIG_PARAMS = "img_processing"
    DEFENSE_CONFIG_PARAMS = "defense"
    TRANS_CONFIG_PARAMS = "trans"
