import dash
import os
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import graphs as gr
import texts as tx

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = '/Bitcoin-Predictor/Dashboard/Static/Descriptions/BM.txt'

# from this file, like this:
text_markdown = "\t"
with open(path) as this_file:
    for a in this_file.read():
        if "\n" in a:
            text_markdown += "\n \t"
        else:
            text_markdown += a

app.layout = html.Div([
                html.Div([
                           dcc.Markdown(text_markdown)
                ])
])


server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')
