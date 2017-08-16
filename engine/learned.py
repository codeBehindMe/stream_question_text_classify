# This module contains functionality serviceable by the machine learned classifier.
import pickle

from .exceptions import InvalidModelException
from engine import IClassifier
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import LancasterStemmer
import numpy as np

# Stop packages from raising warnings to the user.
warnings.filterwarnings("ignore")


class LearnedModelClassifier(IClassifier):
    __MODELS = [SGDClassifier]  # Model type checks.

    def __init__(self, vectoriser, model):
        self._vectoriser = vectoriser
        self._model = model

    def _load_vectoriser(self):
        # type:() -> None

        """
        Flow Control Method
        -------------------
        This method asserts that the loaded vectoriser is of the type that is expected.
        :return: None
        """
        if not isinstance(self._vectoriser, CountVectorizer):
            raise InvalidModelException(
                "Did not receive vectoriser that was of the expected type. Expected sklearn.feature_extraction.text.CountVectorizer. Recieved : {0}".format(
                    type(self._vectoriser)))
        return

    def _load_predictiveModel(self):
        # type: () -> None
        """
        Flow Control Method
        -------------------
        This method asserts that the loaded model is one that is supported by the LearnedModelClassifier class.
        :return: None
        """

        for model in self.__MODELS:
            if isinstance(self.__MODELS, model):
                return

        raise InvalidModelException(
            "Did not recieve a model that was one of the expected types. Use model_types() to see supported models. Recieved type {0}".format(
                type(self._model)))

    @staticmethod
    def _strip_remove_non_alpha(stringSentence):
        # type: (str) -> str
        """
        Removes any whitespace from the end of the string and removes any non alphabetic characters.
        Runtime :: O(N words)
        :param stringSentence: Sentence to be stripped of whitespace and non alpha characters.
        :return: Recompiled string sentence.
        """

        _stripped = stringSentence.strip()
        _words = _stripped.split(' ')
        _realWords = [word for word in _words if word.isalpha()]

        return ' '.join(_realWords)

    @staticmethod
    def _stem_words_lancaster(stringSentence):
        # type: (str) -> str
        """
        Stems a given sentence using the Lancaster stemming algorithm and returns the string.
        Runtime :: O(N words)
        :param stringSentence: Sentence to be stemmed
        :return: Recompiled string sentence.
        """
        _words = stringSentence.split(' ')
        _words = list(map(lambda x: LancasterStemmer().stem(x), _words))

        return ' '.join(_words)

    @classmethod
    def _process_sentence(cls, stringSentence, vectoriser):
        # type: (str,CountVectorizer) -> None
        """
        This method takes a string object and processes so it may be passed into the model object to make a prediction.
        :param stringSentence: String to be processed
        :param vectoriser: vectoriser to be used for feature vector construction.
        :return: sparse matrix containing the feature vector.
        """

        _str = cls._strip_remove_non_alpha(stringSentence)
        _str = cls._stem_words_lancaster(_str)

        _feature_vector = vectoriser.transform(_str)

        raise NotImplementedError()

    def get_supported_models(self):
        # type: () -> list
        """
        This is an attribute method which the supported model types.
        :return:
        """
        return [type(model) for model in self.__MODELS]

    def predict(self):
        raise NotImplementedError()

    def probabilities(self):
        raise NotImplementedError()

    def predict_top(self, n):
        raise NotImplementedError()
