import pickle
from abc import ABCMeta, abstractmethod


class FileIOService:
    @staticmethod
    def loadPickleObject(path, mode_):
        return pickle.load(open(path, mode=mode_))


class IClassifier(metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def probabilities(self):
        pass

    @abstractmethod
    def predict_top(self, n):
        pass