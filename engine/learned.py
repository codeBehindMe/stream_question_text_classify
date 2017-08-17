# This module contains functionality serviceable by the machine learned classifier.

from .exceptions import InvalidModelException
from engine import IClassifier
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import LancasterStemmer
from scipy.sparse import spmatrix

# Stop packages from raising warnings to the user.
warnings.filterwarnings("ignore")


class LearnedModelClassifier(IClassifier):
    __MODELS = [SGDClassifier]  # Model type checks.

    def __init__(self, vectoriser, model, descriptor=None):
        # type: (CountVectorizer,object,dict) -> None
        self._vectoriser = vectoriser
        self._model = model
        self._descriptor = descriptor

        self._load_vectoriser()
        self._load_predictiveModel()

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
        Also assigns the multinomial class labels which can be predicted.
        :return: None
        """

        for model in self.__MODELS:
            if isinstance(self._model, model):
                self._classLabels = self._model.classes_  # Assign the classes available in the classifier to the labels container.
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
        # type: (str,CountVectorizer) -> spmatrix
        """
        This method takes a string object and processes so it may be passed into the model object to make a prediction.
        :param stringSentence: String to be processed
        :param vectoriser: vectoriser to be used for feature vector construction.
        :return: sparse matrix containing the feature vector.
        """

        _str = cls._strip_remove_non_alpha(stringSentence)
        _str = cls._stem_words_lancaster(_str)

        return vectoriser.transform([_str])

    @staticmethod
    def _format_output(predictions, descriptors):
        # type: (list,dict) -> list

        output = []
        for item in predictions:
            output.append(' - '.join([descriptors.get(item[1]), str(item[0]).format('0.00f')]))

        return output

    def get_supported_models(self):
        # type: () -> list
        """
        This is an attribute method which the supported model types.
        :return: List of the supported model types.
        """
        return [model for model in self.__MODELS]

    def predict(self, stringSentence):
        # type: (str) -> str
        """
        Main prediction method to predict the most likely class.
        :param stringSentence: Sentence to be predicted
        :return: Predicted class label.
        """
        _processedString = self._process_sentence(stringSentence, self._vectoriser)
        return self._model.predict(_processedString)

    def probabilities(self):
        raise NotImplementedError()

    def predict_top(self, stringSentence, n=2, descriptive=True):
        # type: (str,int) -> list
        """
        This method returns the top 2 class labels that the model would predict along with the metric used for the prediction.
        :param stringSentence: Sentence to be classified.
        :param n: Number of class labels to be returned.
        :param descriptive: If available show the descriptive class label.
        :return: Tuple of class label and prediction metric.
        """

        # Check top value is the correct index.
        if n > len(self._classLabels):
            raise IndexError("Number of expected results are above the available set of class labels.")

        _probabilities = self._model.predict_proba(self._process_sentence(stringSentence, self._vectoriser))

        if descriptive and self._descriptor is not None:
            return self._format_output(sorted(zip(_probabilities[0], self._classLabels), reverse=True)[:n],
                                       self._descriptor)

        return sorted(zip(_probabilities[0], self._classLabels), reverse=True)[:n]
