# This module contains all items serviceable by the heuristics classifier.

from engine import IClassifier


class HeuristicClassifier(IClassifier):
    def __init__(self):
        pass

    @staticmethod
    def _preprocess(string):
        raise NotImplementedError()

    def predict_top(self, n):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def probabilities(self):
        raise NotImplementedError()
