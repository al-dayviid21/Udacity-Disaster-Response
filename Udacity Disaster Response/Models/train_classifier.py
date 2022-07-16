import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Creates a custom sklearn transformer
    """

    def starting_verb(self, text):
        """
        Creates a starting verb function that tokenizes a sentence and returns a True boolean if its a verb else False
        Parameters: Self, Text
        Returns: Bool
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """
        Fits the input parameters
        Parameters: Features and Labels
        Returns: self
        """
        return self

    def transform(self, X):
        """
        Transform input parameter to provide a panadas dataframe
        Parameters: Features
        Returns: Pandas Dataframe
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Loads database 
    Parameters: Database file path
    Returns: Features, Labels and Label Names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM DisasterResponseTable", engine)
    X = df["message"]
    Y = df.select_dtypes(include=["int64"]).drop("id", axis=1)
    return X, Y, Y.columns


def tokenize(text):
    """
    Text processing
    Parameters: Text
    Returns: A clean token
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Creates a ML pipeline, Highlight a set of parameters, and then use gridsearchcv to find the best parameters.
    Parameters: None
    Return: A pipeline with the best hypertuning parameters
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)), 
            ('tfidf', TfidfTransformer())
        ])),
        
        ('start_verb', StartingVerbExtractor())
    ])), 
    ('mlpc', MultiOutputClassifier(RandomForestClassifier()))
]) 
    parameters = {

        'mlpc__estimator__min_samples_split': [2, 3],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Create predictions and prints out a classifcation report containing the accuracy, precision, recall, f1 score amongst others.
    Parameters: Model, Test data and category names
    Returns: Prints out classification report
    """
    Y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print(classification_report(Y_test.iloc[i], Y_pred[i]))


def save_model(model, model_filepath):
    """
    Saves the ML model
    Parameters: model and the intended model file path
    Returns: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Loads data, build, train, evaluate, and save model
    """
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