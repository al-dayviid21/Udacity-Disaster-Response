import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    The load data function is used to load the data into a pandas dataframe
    Paramters: The data file paths
    Returns: A merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df
    


def clean_data(df):
    """
    The clean data function rids the pandas dataframe of all its quality and tidiness issues
    Parameters: An unclean pandas dataframe
    Returns: A clean dataframe
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    func = lambda x: x[:-2]
    category_colnames = row.apply(func)
    # rename the columns of `categories`
    categories.columns = category_colnames  
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype("str").str.split("-").str[1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    
    categories['related'] = categories.related.replace({2:1})
    # drop the original categories column from `df`
    df = df.drop("categories", axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df

    
def save_data(df, database_filename):
    """
    Saves pandas dataframe into a sqlite database
    Parameters: Pandas dataframe and the database filename
    Returns: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("DisasterResponseTable", engine, index=False, if_exists='replace')  


def main():
    """
    Runs a sequence of functions to load, clean and save a pandas dataframe
    Parameters: None
    Returns: None
    """
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
