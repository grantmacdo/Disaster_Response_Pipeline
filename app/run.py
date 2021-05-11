import json
import plotly
import pandas as pd
import plotly.express as px


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from wordcloud import WordCloud



app = Flask(__name__)

def tokenize(text):
    '''
    input:text
    output: cleaned and tokenized list of the text
    ''' 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/dis_res.db')
df = pd.read_sql_table('InsertTableName', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # find total number of messages in each category
    message_cat_sum = df.drop(df.columns[[0, 1, 2, 3]], axis=1, inplace=False).sum(axis = 0).sort_values(ascending =False)[:10].reset_index()
    message_cat_sum_x = message_cat_sum['index']
    message_cat_sum_y = message_cat_sum[0]
    
    #Graph data extraction: word cloud
    text=' '.join(df.message.tolist())
    token=tokenize(text)
    all_token=' '.join(token)
    wordcloud = WordCloud(max_font_size=70, max_words=200,              background_color="black").generate(all_token)
    
    wordcloud_fig = px.imshow(wordcloud)
    wordcloud_fig.update_layout(
    title= dict(text = '150 Most Common Words', x=0.5),
    xaxis={'showgrid': False, 'showticklabels':False, 'zeroline':False},
    yaxis={'showgrid': False, 'showticklabels':False, 'zeroline':False},
    hovermode=False
    )
    
    
    # TODO: Below is an example - modify to create your own visuals
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
     {
            'data': [
                Bar(
                    x=message_cat_sum_x,
                    y=message_cat_sum_y
                )
            ],

            'layout': {
              'title': 'Top 10 Categories',
               'yaxis': {
                   'title': "Count"
               },
               'xaxis': {
                   'title': "Genre"
               }
           }
       }   
   ]
    
    # adding wordcloud
    graphs.append(wordcloud_fig)
    


   
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()