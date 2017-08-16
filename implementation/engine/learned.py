# This module contains functionality serviceable by the machine learned classifier.
import pickle

from .exceptions import InvalidModelException
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import LancasterStemmer

# Stop packages from raising warnings to the user.
warnings.filterwarnings("ignore")


class LearnedModelClassifier:
    __MODELS = [SGDClassifier()]

    def __init__(self, vectoriser, model):
        self._vectoriser = vectoriser
        self._model = model

    def _load_vectoriser(self):
        # type:() -> None

        """
        Flow control method.
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
        Flow control method.
        This method asserts that the loaded model is one that is supported by the LearnedModelClassifier class.
        :return:
        """

        for model in self.__MODELS:
            if isinstance(self.__MODELS, model):
                return

        raise InvalidModelException(
            "Did not recieve a model that was one of the expected types. Use model_types() to see supported models. Recieved type {0}".format(
                type(self._model)))

    def get_supported_models(self):
        # type: () -> list
        """
        This is an attribute method which the supported model types.
        :return:
        """
        return [type(model) for model in self.__MODELS]
