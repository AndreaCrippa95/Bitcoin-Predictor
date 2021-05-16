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
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'

df = pd.read_csv(path, header=0)

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC Price'
df.columns.values[2] = 'Gold'
df.columns.values[3] = 'NASDAQ'

fig = go.Figure()
fig.add_trace(go.Line(x=df["Date"], y=df["BTC Price"],
                    mode='lines',
                    name=df.columns.values[1]))
fig.add_trace(go.Line(x=df["Date"], y=df["Gold"],
                    mode='lines',
                    name=df.columns.values[2]))
fig.add_trace(go.Line(x=df["Date"], y=df["NASDAQ"],
                    mode='lines',
                    name=df.columns.values[3]))

app.layout = html.Div([
    html.H1(id='text_header', children=' Welcome to our BTC Forecasting Platform.'),

    dcc.Markdown(id='text_description',children='Text description of the project'),

    html.Br(),

    html.Div(["Select commencement date ",
        dcc.DatePickerSingle(
        id='input-date',
        min_date_allowed=date(2010, 1, 1),
        max_date_allowed=date(2021, 1, 1),
        initial_visible_month=date(2015, 4, 5),
        date=date(2030, 8, 25),
        )]),

    html.Br(),
    dcc.Markdown(id='output-date'),

    html.Div(["How many days would you like to predict? ",
              dcc.Input(id='my-input', value='100 days', type='text')]),
    html.Br(),

    html.Div(["How many days would you like to predict? ",
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

    dcc.Graph(id='first_graph',
              figure=fig)

])

server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
