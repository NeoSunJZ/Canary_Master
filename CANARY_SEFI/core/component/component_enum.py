from enum import Enum


class ComponentType(Enum):
    ATTACK = "attack"
    MODEL = "model"
    DEFENSE = "defense"
    TRANS = "trans"


class ComponentConfigHandlerType(Enum):
    ATTACK_PARAMS = "attack_params"
    DEFENSE_PARAMS = "defense_params"
    TRANS_PARAMS = "trans_params"
