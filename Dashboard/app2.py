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

    dcc.DatePickerSingle(
        id='input-date',
        min_date_allowed=date(2010, 1, 1),
        max_date_allowed=date(2021, 1, 1),
        initial_visible_month=date(2015, 4, 5),
        date=date(2017, 8, 25)),

    html.Div(id='output-date'),

    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Enter a number of days and press submit'),

    dcc.Checklist(id='df_checklist', options=[
        {'label': 'Bitcoin Price', 'value': 'BTC'},
        {'label': 'Gold Price', 'value': 'GLD'},
        {'label': 'NASDAQ Index', 'value': 'NASDAQ'}],
        value=['BTC','GLD', 'NASDAQ'])



])

@app.callback(
    [Output('output-date', 'children')],
    [Input('input-date', 'date')])

def update_output(date):
    return ["You have selected the " + str(date)]

@app.callback(
    [Output('container-button-basic', 'children')],
    [Input('submit-val', 'value')])
def update_output(value):
    return 'The input value was "{}" '.format(value)

@app.callback(
    [Output('','')]
    [Input('','')])
def choose_data()

server = app.server
if __name__ == '__main__':
    app.run_server()
