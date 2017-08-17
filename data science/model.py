# Model generation file
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pickle

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

    # Predict
    y_pred = _SGDModel.predict(fv_test)

    # check accuracy
    print(accuracy_score(y_test, y_pred))
    # Accuracy == 0.987654320988
    # Something is not right. No way we can predict every single class label correctly.

    # Try knn classifier
    _KNNModel = KNeighborsClassifier(1).fit(fv_train, y_train)
    _y_pred = _KNNModel.predict(fv_test)

    pickle.dump(_cVectorizer, open("cvectoriser.pobj", "wb"))

    pickle.dump(_SGDModel, open("linearmodel.pobj", "wb"))

    _loadVec = pickle.load(open("cvectoriser.pobj", "rb"))
    _loadModel = pickle.load(open("linearmodel.pobj", "rb"))