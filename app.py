# Author : Aaron Tillekeratne <aaron.tillekeratne@gmail.com>
# Date : 16/08/2017
# Python Version 3.5.2


import logging
import sys
import json
from json import JSONDecodeError

from engine import exceptions
from engine.learned import LearnedModelClassifier
from engine.heuristic import HeuristicClassifier
from engine import FileIOService

if __name__ == '__main__':
    print("starting app")

    classifier = None  # Container for the classifier to be used.
    app_config = None  # Container for application configuration.
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
            if user_input == '1' or user_input == '2':
                user_selection = user_input
            else:
                print("Invalid selection.")
    except:
        raise

    try:
        # Initialise the prediction engine as per users selection.
        if user_selection == '1':
            classifier = LearnedModelClassifier(FileIOService.loadPickleObject(app_config.get('vectoriser'), "rb"),
                                                FileIOService.loadPickleObject(app_config.get('linearmodel'), "rb"))
        elif user_selection == '2':
            classifier = HeuristicClassifier()

    except FileNotFoundError:
        print("Could not find model or vectoriser in the file system.")
    except exceptions.InvalidModelException:
        print("Incorrect model was loaded from the file system.")
