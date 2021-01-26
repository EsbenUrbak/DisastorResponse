# import libraries
import sqlalchemy as db
import pandas as pd
import re
import numpy as np
import scipy.stats as stats
#sklearn libraries
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
#from sklearn.utils.fixes import loguniform
#from sklearn.experimental import enable_halving_search_cv  # noqa
#from sklearn.model_selection import HalvingGridSearchCV

#nltk libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import pickle




def load_data(database_filepath):
    # load data from database
    engine = db.create_engine('sqlite:///Data/'+database_filepath+".db")
    df = pd.read_sql_table(table_name=database_filepath,con=engine)
    #splitting the data into x and y values:
    X = df["message"]  #messages
    df.drop(["message", "original", "id","genre", "index"], axis=1, inplace=True)  #dropping all columns except for the categories
    Y = df
    return X, Y, Y.columns

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    #splitting the sentence into words:
    tokens = word_tokenize(text)

    #secondly, lemmatize the words if the word is not a "stop word"
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    model = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer()),('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame.from_records(y_pred)
    y_pred_pd.columns= Y_test.columns

    #Finding the model stats for each category:
    results_list=[]
    for column in Y_test:
        precision,recall,fscore,support=score(Y_test[column], y_pred_pd[column],average='macro') #,average='macro'
        results_list.append([column,precision, recall, fscore])
    results = pd.DataFrame(results_list, columns = ["category","precision","recall","fscore"])
    print(results)
    accuracy =accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
    print('Accuracy of the model: {}'.format(accuracy))

def save_model(model, model_filepath):
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def model_tuning_grid(model, X_train, Y_train):
    #parameters to be turned:
    parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000),
    #'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [10,100],
    'clf__estimator__max_depth': [5,15],
    'clf__estimator__min_samples_split': [5, 10],
    #'clf__estimator__min_samples_leaf': [2, 5],
    }

    cv = GridSearchCV(model, param_grid=parameters, verbose = 10) #, cv =3
    cv.fit(X_train, Y_train)
    print('Best Score: %s' % cv.best_score_)
    print("\nBest Parameters:", cv.best_params_)
    return cv


def model_tuning_halving(model, X_train, Y_train):
    #parameters to be turned:
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [10, 50,100,300,500],
    'clf__estimator__max_depth': [5, 8,15,25,100],
    'clf__estimator__min_samples_split': [2, 5, 10,15,100],
    'clf__estimator__min_samples_leaf': [1, 2, 5,10],
    }

    cv = HalvingGridSearchCV(model, param_grid=parameters, cv =3, verbose = 5)
    cv.fit(X_train, Y_train)
    print('Best Score: %s' % cv.best_score_)
    print("\nBest Parameters:", cv.best_params_)
    return cv

def main(tune_model = False):
    database_filepath = "DRDB"
    model_filepath = "Model/trained_model.pkl"
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    #best model I have found:
    parameters = {
    #'vect__max_df': 0.75,
    #'vect__max_features': 10,
    #'tfidf__use_idf': True,
    #'clf__estimator__n_estimators': 10,
    #'clf__estimator__max_depth': 15,
    #'clf__estimator__min_samples_split': 10,
    #'clf__estimator__min_samples_leaf': 2,
    }

    print('Building model...')
    model = build_model()
    model.set_params(**parameters)

    print('Training model...')
    if tune_model:
        model = model_tuning_grid(model, X_train, Y_train)
    else:
        model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)


    print('Saving model...\n    MODEL: {}'.format(model_filepath))

    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main(False)
