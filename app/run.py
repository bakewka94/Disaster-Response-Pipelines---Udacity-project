import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data and model
engine = create_engine('sqlite:///../data/DisasterResponse.db')  
df = pd.read_sql_table('messages', engine)  
model = joblib.load("../models/classifier.pkl")  

@app.route('/')
@app.route('/index')
def index():
    # Genre distribution visualization
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Category distribution visualization
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts.index.tolist()
    
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [Bar(x=category_names, y=category_counts)],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category", 'tickangle': -45}
            }
        }
    ]
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '') 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels)) 
    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
