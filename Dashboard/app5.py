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
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor'
sys.path.append(path)

import Inputs

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'

df = pd.read_csv(path, header=0)

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

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

    html.H1(id='text_header', children=' Welcome to our BTC Forecasting Platform.'),

    dcc.Markdown(id='text_description', children='Text description of the project'),

    html.Br(),

    html.Div(["Select commencement date ",
        dcc.DatePickerSingle(
        id='input-date',
        min_date_allowed=Inputs.start,
        max_date_allowed=Inputs.end,
        initial_visible_month=date(2014, 4, 5),
        date=Inputs.start,
        )]),

    html.Br(),

    dcc.Markdown(id='output-date'),

    html.Div(["How many days would you like to predict? ",
              dcc.Input(id='my-input', value='100', type='text')]),
    html.Br(),

    html.Div(["Which indicators would you like to choose, always select BTC ",
              dcc.Checklist(id="checklist_indicator",
                     options=[
                        {'label': 'Bitcoin Price', 'value': 'BTC_Price'},
                        {'label': 'Gold Price', 'value': 'Gold_Price'},
                        {'label': 'NASDAQ Price', 'value': 'NDAQ_Price'}],
                        value=['BTC_Price', 'Gold_Price', 'NDAQ_Price'],
                        labelStyle={'display': 'inline-block'})]),

    html.Br(),

    html.Div(["What model would you like to use ? ",
              dcc.Dropdown(id="dropdown_model",
               options=[
                {'label': 'RandomForestRegressor', 'value': 'RFR'},
                {'label': 'GradientBoostingRegressor', 'value': 'GBR'},
                {'label': 'LinearRegression', 'value': 'LR'},
                {'label': 'Lasso', 'value': 'Lasso'},
                {'label': 'KNeighborsRegressor', 'value': 'KNR'},
                {'label': 'ElasticNet', 'value': 'EN'},
                {'label': 'DecisionTreeRegressor', 'value': 'DTR'}],
               placeholder= " Please do select a model ")]),

    html.Br(),

    html.Div(["This is the actual graph",
              dcc.Graph(id='first_graph',
                figure=fig)]),

    html.Br(),
    html.H1(id='text_results', children= 'Results part.'),

    dcc.Markdown(result_markdown),

    html.Br(),

    dcc.Markdown(accuracy_markdown),

    html.Br(),

])

server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')
