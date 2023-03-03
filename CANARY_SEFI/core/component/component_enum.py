from enum import Enum


class ComponentType(Enum):
    ATTACK = "attack"
    MODEL = "model"
    DEFENSE = "defense"


class ComponentConfigHandlerType(Enum):
    ATTACK_PARAMS = "attack_params"
    DEFENSE_PARAMS = "defense_params"
