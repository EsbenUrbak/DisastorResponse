# import libraries
import sqlalchemy as db
import pandas as pd
import re
import numpy as np
import scipy.stats as stats
from sqlalchemy import create_engine
#sklearn libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

#nltk libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pickle


def load_data(database_filepath):
    """
    loads the sqlite database and returns the x, y values for analysis and the y columns

    parameters
    ----------
        database_filepath: str
            the file path to the database to be loaded

    Returns
    -------
        X: pandas
            messages from the database
        Y: pandas
            the categories in integer format
        Y.columns: pandas
            the names of each (panda)
    """

    # load data from database
    engine = db.create_engine('sqlite:///Data/'+database_filepath+".db")
    df = pd.read_sql_table(table_name=database_filepath,con=engine)
    #splitting the data into x and y values:
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns


def tokenize(text):
    """
    converts the sentences into "tokens" while also removnig punctuation, capitals and lemmatizes the words

    parameters
    ----------
        text: str
            text to be tokenized

    Returns
    -------
        clean_tokens: List
            lower lemmatized tokens as a list
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #splitting the sentence into words:
    tokens = word_tokenize(text)

    #secondly, lemmatize the words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    builds a pipeline with Count Vectorizer and TFIDF first and then a Adaboost classifier

    parameters
    ----------

    Returns
    -------
        model: Sklearn pipeline model
            with Count Vectorizer and TFIDF first and then a Adaboost classifier
    """

    #building the pipeline. Firstly using Count vect and Tfidf to transform the words data into numbers. and then using a Adaboost model.
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(AdaBoostClassifier()))])   #RandomForestClassifier(n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This evaluations the fit of the inputted model vs the actual.

    parameters
    ----------
        model: Sklearn model
            a sklearn model that has been fittet
        X_test:
            The test input values to the model
        Y_test:
            The actual values
        category_names:
            The names of each category

    Returns
    -------
        results: pandas
            a pandas with stast about the fit of the model on the test data
    """

    #predicting using the model:
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame.from_records(y_pred)
    y_pred_pd.columns= Y_test.columns

    #Finding the model stats for each category:
    results_list=[]
    average_accuracy = 0
    for column in Y_test:
        precision,recall,fscore,support=score(Y_test[column], y_pred_pd[column],average='macro') #,average='macro'
        accuracy = accuracy_score(Y_test[column], y_pred_pd[column])
        average_accuracy = average_accuracy + accuracy
        results_list.append([column,precision, recall, fscore, accuracy])
    results = pd.DataFrame(results_list, columns = ["category","precision","recall","fscore", "acccuracy"])
    print(results)
    print('Accuracy {}\n\n'.format(average_accuracy/len(Y_test.columns)))
    return results


def save_model(model, model_filepath):
    """
    This saves the model as a pickle file

    parameters
    ----------
        model: Sklearn model
            a sklearn model that has been fittet
        model_filepath: str
            file path to where one wants the model saved

    Returns
    -------
    """

    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def save_data(df, database_filename):
    """
    This saves the data to a sqlite database

    parameters
    ----------
        df: pandas
            pandas containing the data to be saved
        database_filename: str
            file path to where one wants the data saved

    Returns
    -------
    """
    #saving the model stats
    engine = create_engine('sqlite:///data/'+database_filename+".db")
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def model_tuning_grid(model, X_train, Y_train):
    """
    This builds a GridSearchCV to find the best hyperparameters.

    parameters
    ----------
        model: Sklearn model
            a sklearn model that has been fittet
        X_train:
            The test input values to the model
        Y_train:
            The actual/true values


    Returns
    -------
        cv: Sklearn model
            a fittet and hyperparameters optimized sklearn model
    """

    #parameters to be turned:
    parameters = {'clf__estimator__learning_rate': [0.01, 0.02, 0.05],
              'clf__estimator__n_estimators': [10, 50,100]}
    #setting up the grid searchs
    cv = GridSearchCV(model, param_grid=parameters, verbose = 5, scoring='f1_micro', n_jobs=-1) #, cv =3
    #starting the gridsearch
    cv.fit(X_train, Y_train)
    print('Best Score: %s' % cv.best_score_)
    print("\nBest Parameters:", cv.best_params_)
    return cv


def main(tune_model = False):
    """
    This runs the training of a model on the database data

    parameters
    ----------
        tune_model: Boolean
            if True then it will do a GridSearchCV to find the best hyperparameters
    Returns
    -------

    """

    database_filepath = "DRDB"
    model_stats_filepath = "stats"
    model_filepath = "Model/trained_model.pkl"
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    #best model I have found:
    #0.9486767946436139
    parameters_adaboost = {'clf__estimator__learning_rate': 0.05,'clf__estimator__n_estimators': 100}

    print('Building model...')
    model = build_model()
    model.set_params(**parameters_adaboost)

    print('Training model...')
    if tune_model:
        model = model_tuning_grid(model, X_train, Y_train)
    else:
        model.fit(X_train, Y_train)

    print('Evaluating model...')
    stats = evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model stats data...\n    DATABASE: {}'.format(model_stats_filepath))
    save_data(stats, model_stats_filepath)
    print('stats data saved to database')

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)
    print('Trained model saved!')


if __name__ == '__main__':
    main()
