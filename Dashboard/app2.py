import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children=' Welcome to our BTC Forecasting Platform.'),
    html.Div(["What is your start day",
        dcc.DatePickerSingle(
        id='input-date',
        min_date_allowed=date(2010, 1, 1),
        max_date_allowed=date(2021, 1, 1),
        initial_visible_month=date(2015, 4, 5),
        date=date(2017, 8, 25),
        )]),
    html.Br(),
    html.Div(id='my-output-date'),

    html.H6("How many days would you like to predict?"),
    html.Div(["Number of days",
              dcc.Input(id='my-input', value='initial value', type='text')]),
    html.Br(),
    html.Div(["This is what you choosed",
              dcc.Input(id='my-output')]),

])

@app.callback(
    [Output(component_id='my-output-date',component_property='value')],
    [Input(component_id='input-date', component_property='date')]
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

@app.callback(
    [Output(component_id='my-output',component_property='value')],
    [Input(component_id='my-input',component_property='value')]
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

server = app.server
if __name__ == '__main__':
    app.run_server()

