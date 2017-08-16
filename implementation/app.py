# Author : Aaron Tillekeratne <aaron.tillekeratne@gmail.com>
# Date : 16/08/2017
# Python Version 3.5.2


import logging
import numpy as np
from optparse import OptionParser
import sys
import json
from json import JSONDecodeError

# Option parser
op = OptionParser()

op.add_option("--learned", dest="model_predict")
op.add_option("--heuristic", dest="heuristic_predict")

if __name__ == '__main__':
    print("starting app")

    try:
        # Load configuration
        with open('config.json') as file:
            app_config = json.load(file)
        file.close()
    except FileNotFoundError:
        print("Could not find config.json file.")
    except JSONDecodeError:
        print("config.json file has been corrupted.")
