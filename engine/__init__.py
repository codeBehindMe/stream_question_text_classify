import pickle
from abc import ABCMeta, abstractmethod


class FileIOService:
    @staticmethod
    def load_pickle_obj(path, mode_):
        return pickle.load(open(path, mode=mode_))


class IClassifier(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, stringSentence):
        pass

    @abstractmethod
    def probabilities(self):
        pass

    @abstractmethod
    def predict_top(self,stringSentence, n):
        pass
