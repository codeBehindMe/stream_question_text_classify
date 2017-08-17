# Model generation file
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import pickle
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Read in the data
    _data = pd.read_csv("CLEANED_LABELED.csv")

    # Tokenize and vectorise.
    _cVectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 3))
    # Make test and training data.
    x_train, x_test, y_train, y_test = train_test_split(_data['question_text'], _data['code'], test_size=0.3,
                                                        random_state=300)
    # Fit the vectoriser using training data.
    _cVectorizer.fit(x_train)

    # Generate feature vectors
    fv_train = _cVectorizer.transform(x_train)
    fv_test = _cVectorizer.transform(x_test)

    # Logistic regression mode with default parameters
    _SGDModel = SGDClassifier(loss="log").fit(fv_train, y_train)

    pickle.dump(_cVectorizer, open("cvectoriser.pobj", "wb"))
    pickle.dump(_SGDModel, open("linearmodel.pobj", "wb"))
