from abc import ABCMeta
from abc import abstractmethod


class Attack(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def perturbation(self):
        print("Abstract Method of Attacks is not implemented")
        raise NotImplementedError