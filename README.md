
# Disaster Response Pipeline Project

## Table of Contents
1. Installation
2. Project Summary
3. File Descriptions
4. Instructions
5. Licensing, Authors, and Acknowledgements

## Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Summary
This project is a web app where an emergency worker can input a new message that were sent during disaster events and get classification results in several categories so that they can be sent an appropriate disaster relief agency. The web app will also display visualizations of the data pecularities. 

## File Descriptions
The file structure for the project with descriptions are shown below:
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Licensing, Authors, and Acknowledgements
Credit is due Appen for the dataset and Udacity for their guide during the formation of project codes.
