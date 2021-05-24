import dash
import base64
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

import graphs as gr

file_path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(file_path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

prediction_days = 100
Real_Price = df.loc[:,'BTC_Price']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = go.Figure()
fig.add_trace(go.Line(x=df["Date"], y=df["BTC_Price"],
                    mode='lines',
                    name=df.columns.values[1]))
fig.add_trace(go.Line(x=df["Date"], y=df["Gold_Price"],
                    mode='lines',
                    name=df.columns.values[2]))
fig.add_trace(go.Line(x=df["Date"], y=df["NDAQ_Price"],
                    mode='lines',
                    name=df.columns.values[3]))

path_result = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/Result.txt'
result_markdown = "\t"
with open('/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/Result.txt') as this_file:
    for a in this_file.read():
        if "\n" in a:
            result_markdown += "\n"
        else:
            result_markdown += a

accuracy_markdown = "\t"
with open('/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/Accuracy.txt') as this_file:
    for a in this_file.read():
        if "\n" in a:
            accuracy_markdown += "\n \t"
        else:
            accuracy_markdown += a

app.layout = html.Div(
                    children=[

    html.H1(id='text_header', children='Welcome to our Dashboard.'),

    dcc.Markdown(id='text_description', children=' Andrea ! Maximilian ! Flavio '),

    html.Br(),

    html.Div(["This is the actual graph",
              dcc.Graph(id='first_graph',
                figure=fig)]),

    html.Br(),
    html.H1(id='text_results', children= 'Results part.'),

    dcc.Markdown(result_markdown),

    html.Img(src='data:image/png;base64,{}'.format(gr.BM_png64)),

    html.Br(),

    dcc.Markdown(accuracy_markdown),


])

server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')
