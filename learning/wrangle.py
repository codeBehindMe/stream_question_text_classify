# This file is for exploring and wrangling the preprocessed data.
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams


def getTokensFromString(string, regexPattern=r'\W+'):
    # type: (str,str) -> list
    """
    This method returns a list of tokens separated by whitespace.
    :param string: String to be tokenized.
    :param regexPattern: Regular expression pattern to tokenize on. Defaults to whitespace.
    :return: List containing the tokens.
    """

    _tokenizer = RegexpTokenizer(regexPattern, gaps=True)
    return _tokenizer.tokenize(string)


if __name__ == '__main__':
    # read in the cleaned data.
    _data = pd.read_csv("PROCESSED_LABELED.csv")

    # The balance of classes are important for a learning algorithm, so let's quickly visualise that.
    _data.groupby(['code']).count().reset_index().plot(kind='bar', x='code')
    # The classes are not completely biased, but some class labels are more prominent than others.
    #     code  question_text
    # 0  ALI.5             63
    # 1  ENA.3             60
    # 2  INN.2             74
    # 3  TEA.2            100

    # Before we head out let's cast everything to lower case.
    _data['question_text'] = _data['question_text'].str.lower()

    # Being realistic about time, we won't be able to do anything fancy. So let's take a bag of words approach to this. Lets start with some simple whitespace tokenization to build our bag of words.
    _wordBag = map(lambda x: getTokensFromString(x), _data['question_text'].tolist())

    # Flatten into one list. This method actually seems to be the fastest. =\\
    _flatBag = [word for wordList in list(_wordBag) for word in wordList]

    # Remove any numbers that might be present.
    _flatBag = [word for word in _flatBag if not word.isdigit()]

    # Let's generate some bigrams
    # TODO: Bigrams