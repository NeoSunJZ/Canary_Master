from enum import Enum


class ComponentType(Enum):
    ATTACK = "attack"
    MODEL = "model"


class ComponentConfigHandlerType(Enum):
    ATTACK_PARAMS = "attack_params"
