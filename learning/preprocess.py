# This file contains the data preprocessing for the learning stage.
# This could be done nicer with a jupyter notebook.
import pandas as pd
import os


def checkDataReadHealth(data):
    # type: (pd.DataFrame) -> None
    with open("../original_files/labeled_data.csv") as file:
        _data = file.readlines()
    assert _originalLabelled.shape[0] == (len(_data) - 1)


def cleanUpCodeColumn(originalData):
    # type: (pd.DataFrame) -> pd.DataFrame

    """
    Cleans the code column and the dataframe by:
    1. Renaming the code column to remove the commas.
    2. Removing whitespace and commas from the code column values.
    :param originalData: Pandas dataframe containing the original data.
    :return: Cleaned up dataframe.
    """

    _local = originalData.copy()
    _local.columns = ['question_text', 'code']

    _local['code'] = _local['code'].apply(lambda x: str(x).split(',')[0].strip())

    assert originalData.shape == _local.shape

    return _local


if __name__ == '__main__':

    # Read in the data.
    # Looks like implicitly loading is causing an encoding error.
    # We know that its pipe separated. I tried the most common encodings, and iso-8859-1 seems to work.
    _originalLabelled = pd.read_csv("../original_files/labeled_data.csv", sep="|", encoding='iso-8859-1', dtype=object)

    # Double check that we read in all the data
    # Let's use stack space to keep the variables in scope clean.
    checkDataReadHealth(_originalLabelled)

    # Have a look at the head
    _originalLabelled.head()

    # question_text      code,,
    # 0  I know what my goals are and what I need to do...  ALI.5   ,,
    # 1         I feel like I can be successful in my role  ALI.5   ,,
    # 2  30. I know what I need to do to be successful ...  ALI.5   ,,
    # 3    I understand my role and what is expected of me  ALI.5   ,,
    # 4          I know what is expected of me in my role.  ALI.5   ,,

    # Seems like the code column can use some cleaning. Let's do that

    _cleanLabel = cleanUpCodeColumn(_originalLabelled)
    # Let's look at the distinct values in the code column.
    _cleanLabel['code'].unique()
    # array(['ALI.5', 'nan', 'ENA.3', 'TEA.2', 'INN.2'], dtype=object)

    # Looks like theres a nan in there. Let's get rid of the missing row.
    _cleanLabel = _cleanLabel.loc[_cleanLabel['code'] != 'nan']

    # Let's write out a cleaner output file for data the learning process.
    _cleanLabel.to_csv("../learning/CLEANED_LABELED.csv", encoding='utf-8', index=False)
