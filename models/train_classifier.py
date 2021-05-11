import sys
import sqlite3
import pandas as pd
from sqlalchemy import *
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.model_selection import GridSearchCV

from statistics import mean

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

import pickle
import time

def load_data(database_filepath):
    ''' 
    input : database path
    output : X, Y to use in ML
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    #engine.table_names()
    df = pd.read_sql_table('InsertTableName', engine)
    X = df.message 
    Y = df.iloc[:, 4:]
    category_names = list(df.columns)
    return X, Y, category_names


def tokenize(text):
    '''
    input:text
    output: cleaned and tokenized list of the text
    ''' 
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok =  lemmatizer.lemmatize( (tok.strip()).lower())
        
        
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    output: model
    '''
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    

    
    parameters = {
        'clf__estimator__min_samples_split': [2]
    }

    model = GridSearchCV(pipeline,param_grid=parameters, n_jobs=2, verbose=3)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    output: prints f1 score, precision score and recall score for all categories. 
    An overall accuracy is printed showing the proportion of correct predictions. 
    '''
    y_pred = model.predict(X_test)
    
    accuracy=[]
    for a,b,c in zip(Y_test, y_pred.T, category_names):  
        print("Column Name: "+c)
        print("f1 score: {}".format(f1_score(Y_test[a], b, average = None)))
        print("Precision score: {}".format(precision_score(Y_test[a], b, average = None)))
        print("Recall score: {}".format(recall_score(Y_test[a], b, average = None)))
        print()
        accuracy.append((Y_test[a] == b).mean())
    
    
    print('Mean accuracy')
    print(mean(accuracy))
    pass


def save_model(model, model_filepath):
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()