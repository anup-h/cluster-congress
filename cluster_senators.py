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


parties = pd.read_csv("party_labels.csv")

labels_and_numbers = pd.read_csv("scores.csv").drop(columns = ['Unnamed: 0'])
labels_and_numbers["Party"] = parties["Label"]
df = labels_and_numbers.drop(columns = ["Name", "Handle", "Party"])
df = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
feature_names = df.columns
pca = PCA(n_components=2).fit(df)
df = pca.transform(df)
# print(df)
for i in range(len(pca.singular_values_)):
    df[:, i] = df[:,  i]/pca.singular_values_[i]

kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
labels_and_numbers["Group"] = kmeans.labels_

# print(labels_and_numbers)



app.layout = html.Div(children=[
    html.H1(children='PolarBears'),
    
    html.Div([dcc.Graph(id='main_scatter'), dcc.Slider(
        id='clustering',
        min=1,
        max=5,
        value=1,
        marks={str(d): d if d>1 else "Party" for d in range(1,6)}
    )], style={'marginLeft': 50, 'marginRight': 500})
    
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
            margin=go.layout.Margin(
                l=0,
                r=300,
                b=0,
                t=0,
                pad=4
            ),
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)