# Author : Aaron Tillekeratne <aaron.tillekeratne@gmail.com>
# Date : 16/08/2017
# Python Version 3.5.2


import logging
import numpy as np
from optparse import OptionParser
import sys
import json
from json import JSONDecodeError

from engine import exceptions
from engine import learned

if __name__ == '__main__':
    print("starting app")

    classifier = None  # Container for the classifier to be used.

    try:
        # Load configuration
        with open('config.json') as file:
            app_config = json.load(file)
        file.close()
    except FileNotFoundError:
        print("Could not find config.json file.")
    except JSONDecodeError:
        print("config.json file has been corrupted.")

    try:
        # Ask the user to select a classifier.

        user_selection = None
        while user_selection is None:
            print("Select a classifier to be used for sentence classification.")
            print("1 Linear model")
            print("2 Edit distance")
            user_input = input()


    except:
        raise
