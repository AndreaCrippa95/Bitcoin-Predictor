import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date

from dash import Dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/data/DF_train.csv'
df = pd.read_csv(path, header=0)
df.head()
df.columns.values[0] = 'Date'
df.columns.values[2] = 'Gold'
df.columns.values[3] = 'NASDAQ'

prediction_days = 100

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    dcc.Dropdown(id="dropdown_model",
            options=[
                {'label': 'RandomForestRegressor', 'value': 'RFR'},
                {'label': 'GradientBoostingRegressor', 'value': 'GBR'},
                {'label': 'LinearRegression', 'value': 'LR'},
                {'label': 'Lasso', 'value': 'Lasso'},
                {'label': 'KNeighborsRegressor', 'value': 'KNR'},
                {'label': 'ElasticNet', 'value': 'EN'},
                {'label': 'DecisionTreeRegressor', 'value': 'DTR'}],
            placeholder="Select a model"),

    html.Div(id='id-output')

])

@app.callback(
    [Output(component_id='id-output',component_property='children')],
    [Input(component_id='dropdown_model',component_property='value')])


server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
