# Installation

  - pandas
  - re
  - sys
  - json
  - sklearn
  - nltk
  - sqlalchemy
  - pickle
  - Flask
  - plotly
  - sqlite3
  - joblib
  - wordcloud

# Project Overview

In this project, I analyze disaster messages provided from Figure Eight https://appen.com/ and build a web app that classifies messages using a ML pipeline into 36 categories.

This classification could potentially help so that a message sent during a disaster could be effectively redirected to the appropriate disaster relief agency. The dataset contains 30,000 messages taken from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

In the web app that is build with this training data one could enter any disaster related message and a classification to related categories will be given to the user as output.

## File Descriptions

- Data
  - disaster_messages.csv: CSV file from figure8 for messages.
  - disaster_categories.csv: CSV file from figure8 for categories.
  - process_data.py: Python script to clean and create a database.
  - ETL Pipeline Preparation.ipynb:  Step by step notebook version of process_data
  - dis_res: Database created via the script.
- Model
  - train_classifier.py: Python script of ML pipeline.
  - ML Pipeline Preparation.ipynb: Step by step notebook version of train_classifier.
- App
  - run.py: Python script to run web app



## How to run

You can run following commands in the project's directory. (cd to root file path)

  1. To run pipeline for cleaning data and storing in database
$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/dis_res.db
  2. To run ML pipeline that trains and saves the classifier 
$ python models/train_classifier.py data/dis_res.db models/model.pkl
  3. To run web app cd to app directory and execute
$ python run.py
  4. To open web app go to http://0.0.0.0:3001/