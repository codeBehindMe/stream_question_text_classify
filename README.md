# README #

### What's this repository for? ###

This repository contains the problem and solution to the CAmper initiation challenge. 

### What's inside? ###

On the face of it, there are a number of files. But the main application and dependancies sit in the root folder along with the /engine python module. Find a more detailed analysis of the application below.
The 'data science' folder contains ipython notebooks and python script files that have been used to do data science. This includes things such as pre-processing, wrangling, EDA and modelling.

### How do I get set up? ###

This application is written in python 3.5.2 I have purposefully chosen not to write a setup.bat file as to avoid generating unecessary environments. 
But if you would like to do so, following are the dependancies that you will need.
#### Dependencies ####
* json
* pandas
* sklearn
* pickle
* warnings
* nltk

To start the app, simply call the app.py file from the terminal (python app.py).  

### What is the rationale and methodology in the design of this app? ###

The aim is to build a relatively lightweight application which can classify input strings from the commandline. You can enter any string and it will return the closes reference question along with the probability that the input string belonging to that class, as outlined in the problem statement.

The applications configuraion is stored in a 'config.json' file. Inside this file there are three objects. First two containing the pointers on where to find the vectoriser and the model for the applications classification engine.
The third object contains a dictionary of class labels and the reference questions belonging to that label. Having the models this way helps us scale the applications to more models, vectorisers and even in the case where we need to change the output description of the class labels, without having to rewrite program code.

#### app.py ####
This file contains the main body of the program. It contains four main parts, separated by try catch blocks. 
In the first block contains the wind up sequence, which essentially loads the config file. This can also be extended to do other wind up tasks.
The second and third blocks are implemented for exensibility, such as integrating a different model. These blocks are currently only executes three lines which is hidden from the user.
The last block is the main loop of the program which does the UI handling. It takes the string input and passes it to the application engine which does the actual prediction.

#### engine ####
The engine library delivers the actual heavy lifting of the application. In its declaration, it embeds two system level classes. 
FileIOService: 
This class handles the interface between the file system. Current implementation holds one public method.

### Who do I talk to? ###

* Aaron