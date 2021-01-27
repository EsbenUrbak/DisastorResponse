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
    # load data from database
    engine = db.create_engine('sqlite:///Data/'+database_filepath+".db")
    df = pd.read_sql_table(table_name=database_filepath,con=engine)
    #splitting the data into x and y values:
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns

def tokenize(text):
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
    #building the pipeline. Firstly using Count vect and Tfidf to transform the words data into numbers. and then using a Adaboost model.
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(AdaBoostClassifier()))])   #RandomForestClassifier(n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
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
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

def save_data(df, database_filename):
    #saving the model stats
    engine = create_engine('sqlite:///data/'+database_filename+".db")
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def model_tuning_grid(model, X_train, Y_train):
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
    ######
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
    main(True)
