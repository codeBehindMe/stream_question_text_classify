# Author : Aaron Tillekeratne <aaron.tillekeratne@gmail.com>
# Date : 16/08/2017
# Python Version 3.5.2

# TODO: Implement unit tests

import json
from json import JSONDecodeError

from engine import exceptions
from engine.learned import LearnedModelClassifier
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
        # This block is embedded for extensibility.
        # Default the user selection to the linear model
        user_selection = 1
        while user_selection is None:
            print("Select a classifier to be used for sentence classification.")
            print("1 Linear model")
            print("2 Edit distance")
            user_input = input()
            if user_input == '1' or user_input == '2':
                user_selection = int(user_input)
            else:
                print("Invalid selection.")
    except:
        raise

    try:
        # This block is added for extensibility
        # Initialise the classifier.
        if user_selection == 1:
            classifier = LearnedModelClassifier(
                vectoriser=FileIOService.load_pickle_obj(app_config.get('vectoriser'), "rb"),
                model=FileIOService.load_pickle_obj(app_config.get('linearmodel'), "rb"),
                descriptor=app_config.get("classkeys"))
    except FileNotFoundError:
        print("Could not find model or vectoriser in the file system.")
    except exceptions.InvalidModelException as e:
        print("Problem loading models: " + e.message)

    # main loop of the application

    try:
        while True:
            input_sentence = input("Input a string to be classified: \n")
            print(classifier.predict_top(input_sentence))
    except:
        raise
