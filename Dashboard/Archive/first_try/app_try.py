import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

import os
import sys
path = '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor'
sys.path.append(path)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(path, header=0)

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=date(1995, 8, 5),
        max_date_allowed=date(2017, 9, 19),
        initial_visible_month=date(2017, 8, 5),
        date=date(2017, 8, 25)
    ),
    html.Div(id='output-container-date-picker-single'),

    html.Button('Button 1', id='btn-nclicks-1', n_clicks=0)
])


@app.callback(
    [Output('output-container-date-picker-single', 'children')],
    [Input('my-date-picker-single', 'date')])
def update_output(date_value):
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        return date_string

@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks'))
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        return os.popen(path + '/Shield.sh')

server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')
