import numpy as np
from sklearn.cluster import KMeans
from sklearn import  preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server 

party_dict = {'0': 'Democrat', '1': 'Republican', '2': 'Independent', '3': 'President'}

parties = pd.read_csv("party_labels.csv")

labels_and_numbers = pd.read_csv("scores.csv").drop(columns = ['Unnamed: 0'])
labels_and_numbers["Party"] = parties["Label"]
labels_and_numbers["Name"] = parties["Name"]
df = labels_and_numbers.drop(columns = ["Name", "Handle", "Party"])
df = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
feature_names = df.columns
pca = PCA(n_components=2).fit(df)
df = pca.transform(df)
for i in range(len(pca.singular_values_)):
    df[:, i] = df[:,  i]/pca.singular_values_[i]

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
labels_and_numbers["Group"] = kmeans.labels_

app.layout = html.Div(children=[
    html.H1('PolarBears'),
    html.Div([
        html.P("""With election season just around the corner, we thought it’d be interesting to analyze the political 
        stances of various US politicians without the biases of traditional political views. To do this, we performed 
        sentiment analysis on the tweet histories of all US senators, specifically focusing on tweets that contained 
        contentious keywords. Then, using unsupervised machine learning methods we transformed and clustered the data. 
        The results are displayed here, with different colors representing the different clusters each point belongs to. 
        Feel free to play around!"""),
        dcc.Graph(id='main_scatter'), 
        html.P(children = 'Click on a senator to see data about their political views', id = 'senator_info'),
        html.P(children = '', id = 'top_five'),
        html.P(children = '', id = 'bottom_five'),
        html.P(children = '', id = 'nearest_neighbors'),
        html.H4('Number of Clusters:'),
        dcc.Slider(
            id='clustering',
            min=1,
            max=5,
            value=1,
            marks={str(d): d if d>1 else "Party Affiliations" for d in range(1,6)}
        ),
        ], style={'marginLeft': 150, 'marginRight': 450})

    
])

@app.callback(
    dash.dependencies.Output('main_scatter', 'figure'),
    [dash.dependencies.Input('clustering', 'value')])
def update_graph(clusters):
    if clusters==1:
        labels_and_numbers["Group"] = parties["Label"]
        c = ["blue" if i == 0 else ("red" if i==1 else ("green" if i==2 else "orange"))  for i in labels_and_numbers["Party"]]
    else:
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(df)
        labels_and_numbers["Group"] = kmeans.labels_
        c = kmeans.labels_
    return {
        'data': [
            go.Scatter(
                x = df[:, 0],
                y = df[:, 1],
                mode = 'markers',
                text = labels_and_numbers["Name"],
                hoverinfo = 'text',
                marker= dict(
                    color = c,
                    colorscale= "Viridis"
                ),
                hoverlabel= dict(
                    bgcolor = ["blue" if i == 0 else ("red" if i==1 else ("green" if i==2 else "orange"))  for i in labels_and_numbers["Party"]]
                )
            )
        ],
        'layout': go.Layout(
            hovermode='closest',
        )
    }

@app.callback(
    dash.dependencies.Output('senator_info', 'children'),
    [dash.dependencies.Input('main_scatter', 'clickData')])
def senator_info(data):
    if data:
        index = data['points'][0]['pointIndex']
        row = {k:v[index] for (k, v) in labels_and_numbers.iloc[[index]].to_dict().items()}
        return str(row["Name"]+": "+ party_dict[str(row["Party"])])
    return 'Click on a senator to see data about their political views'

@app.callback(
    dash.dependencies.Output('top_five', 'children'),
    [dash.dependencies.Input('main_scatter', 'clickData')])
def top_five(data):
    if data:
        index = data['points'][0]['pointIndex']
        row = {k:v[index] for (k, v) in labels_and_numbers.iloc[[index]].to_dict().items()}
        data = [(i, row[i]) for  i in row.keys() if not any([i=="Name", i=="Party", i=="Group", i=="Handle"])]
        data.sort(key = lambda x: x[1], reverse = True)
        return "Top five positive keywords: "+ data[0][0] + ", "+ data[1][0] + ", "+ data[2][0] + ", "+ data[3][0] + ", "+ data[4][0]
    return ''

@app.callback(
    dash.dependencies.Output('bottom_five', 'children'),
    [dash.dependencies.Input('main_scatter', 'clickData')])
def bottom_five(data):
    if data:
        index = data['points'][0]['pointIndex']
        row = {k:v[index] for (k, v) in labels_and_numbers.iloc[[index]].to_dict().items()}
        data = [(i, row[i]) for  i in row.keys() if not any([i=="Name", i=="Party", i=="Group", i=="Handle"])]
        data.sort(key = lambda x: x[1])
        return "Top five negative keywords: "+ data[0][0] + ", "+ data[1][0] + ", "+ data[2][0] + ", "+ data[3][0] + ", "+ data[4][0]
    return ''

@app.callback(
    dash.dependencies.Output('nearest_neighbors', 'children'),
    [dash.dependencies.Input('main_scatter', 'clickData')])
def nearest_neighbors(data):
    if data:
        lst = []
        index = data['points'][0]['pointIndex']
        pca_data = df[index]
        for r in range(len(labels_and_numbers)):
            if(r != index):
                other_pca_data = df[r]
                lst.append((r, np.linalg.norm(pca_data-other_pca_data)))
        lst.sort(key = lambda x: x[1])
        data = [str(labels_and_numbers.iloc[[elem[0]]]["Name"][elem[0]]) for elem in lst]
        return "Closest in political stance: "+ data[0] + ", "+ data[1] + ", "+ data[2] + ", "+ data[3] + ", "+ data[4]
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
