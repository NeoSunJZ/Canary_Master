from collections import UserDict
from copy import deepcopy
from enum import Enum

from CANARY_SEFI.core.component.component_exception import ComponentNotFindError, SubComponentNotFindError


def add_dict(dict_1, dict_2):
    dic = deepcopy(dict_1)
    for key in dict_2.keys():
        if key in dic:
            dic[key].update(dict_2[key])
        else:
            dic[key] = dict_2[key]
    return dic


class ComponentDictType(Enum):
    ComponentDict = "component_dict"
    SubComponentDict = "sub_component_dict"


class ComponentDict(UserDict):
    def __init__(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        if args:
            dict = args[0]
            super().__init__(dict)
        elif 'dict' in kwargs:
            dict = kwargs.pop('dict')
            super().__init__(dict)

        self.dict_type = kwargs.get("dict_type")
        if type(self.dict_type) is not ComponentDictType:
            raise TypeError("[SEFI SYSTEM] Initializing ComponentDict USED WRONG ComponentDictType")

        self.component_name = kwargs.get("component_name", None)
        self.component_type = kwargs.get("component_type", None)

    def __missing__(self, key):
        if self.dict_type == ComponentDictType.ComponentDict:
            raise ComponentNotFindError(component_name=key, component_type=self.component_type)
        elif self.dict_type == ComponentDictType.SubComponentDict:
            raise SubComponentNotFindError(sub_component_name=key, component_name=self.component_name,
                                           component_type=self.component_type)

    def get(self, key, default=None, allow_not_exist=False):
        if allow_not_exist:
            if key in self.data:
                return self[key]
            else:
                return default
        else:
            return self[key]
