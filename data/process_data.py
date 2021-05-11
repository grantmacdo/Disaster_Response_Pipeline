# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input: file paths of csv files
    output: creates combined data frame csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge both datesets together
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    input: combined dataframe
    output: cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories_36 = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories_36.loc[0]
    # extract list of new column names for categories using
    # lambda function that takes everything 
    # up to the second to last character of each string
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of `categories_36`
    categories_36.columns = category_colnames
    #convert categories_36 columns to binary values
    for column in categories_36:
        # set each value to be the last character of the string
        categories_36[column] = categories_36[column].str[-1]
        # convert column from string to numeric
        categories_36[column] = categories_36[column].astype(int)
    
    #'related' column has some '2' values, replace with mode value (1)
    categories_36.loc[(categories_36.related == 2), 'related'] = 1    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace = True)
    
    # concatenate the original dataframe with the new `categories_36` dataframe
    df = pd.concat([df, categories_36], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    input: previously created df and file name for the database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()