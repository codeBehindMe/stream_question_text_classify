# This file contains the data preprocessing for the learning stage.
import pandas as pd


def checkDataReadHealth(data):
    # type: (pd.DataFrame) -> None
    """
    Checks to see that the pandas object correctly read all lines.
    :param data: pandas data frame containing read data.
    :return: None
    """
    with open("./original_files/labeled_data.csv") as file:
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
    _originalLabelled = pd.read_csv("./original_files/labeled_data.csv", sep="|", encoding='iso-8859-1', dtype=object)

    # Double check that we read in all the data
    checkDataReadHealth(_originalLabelled)

    # Seems like the code column can use some cleaning. Let's do that
    _cleanLabel = cleanUpCodeColumn(_originalLabelled)
    # Remove nan line
    _cleanLabel = _cleanLabel.loc[_cleanLabel['code'] != 'nan']

    # Let's write out a cleaner output file for data the learning process.
    _cleanLabel.to_csv("./PROCESSED_LABELED.csv", encoding='utf-8', index=False)
