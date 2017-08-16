# This file is for exploring and wrangling the preprocessed data.
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk import FreqDist
import matplotlib.pyplot as plt
from nltk.tokenize import MWETokenizer
from nltk.stem import LancasterStemmer


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


# MWE tokenizer wrapper for generating a tokenizer with identified ngrams.
def getMWETokenizer(nGrams):
    # type: (list) -> MWETokenizer
    """
    We need to generate a multi word expression tokenizer using ngrams that have been identified.
    :param nGrams: List containing nltk.probability.FreqDist objects.
    :return: Tokenizer object.
    """
    _tk = MWETokenizer(separator=' ')
    for i in nGrams:
        _tk.add_mwe(i[0])

    return _tk


# Wrapper for generating tokens with a tokenizer that has ngrams defined.
def getMWETokens(string, tokenizer):
    # type: (str,MWETokenizer) -> list
    """
    This wrapper will generate tokens for a given string using the the specified MWETokenizer.
    :param string: String to be tokenized.
    :param tokenizer: Tokenizer to be used for tokenization.
    :return: list of tokenized strings.
    """
    return tokenizer.tokenize(string.split())


def removeNonAlphabetic(stringSentence):
    # type: (str) -> str
    """
    This method takes a string sentance and removes any characters that are non alphabetic.
    :param stringSentence: Sentence to be processed.
    :return:
    """
    _tokens = stringSentence.split(' ')
    _realWords = [word for word in _tokens if word.isalpha()]

    return ' '.join(_realWords)


def stemWords(stringSentence, stemmer):
    # type: (str,object) -> str
    """
    Stems words using a stemmer.
    :param stringSentence: Sentence to be processed.
    :return:
    """

    _tokens = stringSentence.split(' ')

    _tokens = list(map(lambda x: stemmer.stem(x), _tokens))

    return ' '.join(_tokens)


if __name__ == '__main__':
    # read in the cleaned data.
    _data = pd.read_csv("learning/PROCESSED_LABELED.csv")

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

    _unigrams = FreqDist(ngrams(_flatBag, n=1))
    _unigrams.plot(30)

    # Let's generate some bigrams
    _bigrams = FreqDist(ngrams(_flatBag, n=2))
    _bigrams.plot(30)
    # It's not so much a sharp drop, so maybe stopwords aren't really a problme?
    # Try trigrams
    _trigrams = FreqDist(ngrams(_flatBag, n=3))
    _trigrams.plot(30)

    # Generate tokenizer using bigrams and trigrams top say 10.
    _mweTokenizer = getMWETokenizer(_bigrams.most_common(10) + _trigrams.most_common(10))
    _mweWordBag = map(lambda x: getMWETokens(x, _mweTokenizer), _data['question_text'].tolist())

    # flatten
    _mweWordBagFlat = [word for wordList in list(_mweWordBag) for word in wordList]

    # Let's have a look at some duplicate entries in questions.
    r_duplicates = _data.drop_duplicates()
    print(r_duplicates.shape[0], _data.shape[0])
    # 288 297
    # Looks like 9 instances were duplicates.

    # Remove the non alpha characters.
    r_duplicates['question_text'] = r_duplicates['question_text'].apply(lambda x: removeNonAlphabetic(x))
    # Stem words
    _stemmer = LancasterStemmer()
    r_duplicates['question_text'] = r_duplicates['question_text'].apply(lambda x: stemWords(x, _stemmer))

    r_duplicates = r_duplicates.drop_duplicates()
    print(r_duplicates.shape[0], _data.shape[0])

    r_duplicates.to_csv("../learning/CLEANED_LABELED.csv",index=False,encoding='utf-8')