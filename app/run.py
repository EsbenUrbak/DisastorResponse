

import json
import plotly
import pandas as pd

import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib as joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

    # tokens = word_tokenize(text)
    # lemmatizer = WordNetLemmatizer()
    #
    # clean_tokens = []
    # for tok in tokens:
    #     clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    #     clean_tokens.append(clean_tok)
    #
    # return clean_tokens

# load data
database_filepath = "DRDB"
model_filepath = "Model/trained_model.pkl"
stats = "stats"

engine = create_engine('sqlite:///Data/'+database_filepath+".db")
df = pd.read_sql_table(table_name=database_filepath,con=engine)

engine = create_engine('sqlite:///Data/'+stats+".db")
df_stats = pd.read_sql_table(table_name=stats,con=engine)


# load model
model = joblib.load(model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_names = df.iloc[:,4:].columns
    category_bool = (df.iloc[:,4:] != 0).sum().values
    stats_names = df_stats["category"]
    stats_bool = df_stats["acccuracy"]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        #graph with number of values per category
        {
            'data': [
                Bar(
                    x= category_names,
                    y= category_bool
                )
            ],
            'layout': {
                'title': 'Distribution of Categories across Messages',
                'yaxis':{
                    'title':"Count"
                },
                'xaxis': {
                    'title':"Categories"

                }
            }
        },
        #graph showing the accuracy by category_bool
                #graph with number of values per category
        {
            'data': [
                Bar(
                    x= stats_names,
                    y= stats_bool*100
                )
            ],
            'layout': {
                'title': 'Model Accuracy per category ',
                'yaxis':{
                    'title':"Accuracy (%)"
                },
                'xaxis': {
                    'title':"Categories"

                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run() #host='0.0.0.0', port=3001, debug=True


if __name__ == '__main__':
    main()
