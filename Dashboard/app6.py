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
import subprocess

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'

df = pd.read_csv('data/DataFrame', header=0)

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
        date=date(2021, 1, 1),
        )]),
    html.Div(id='output-date'),
    html.Br(),


    html.Div(["How many days would you like to predict? ",
              dcc.Input(id='my-input', value='100 days', type='text')]),
    html.Div(id="output-days"),
    html.Br(),

    html.Div(["What model would you like to choose?",
              dcc.Dropdown(id="dropdown_model",
               options=[
                 {'label': 'BrownianMotion', 'value': 'BM'},
                 {'label': 'RandomForestRegressor', 'value': 'RFR'},
                 {'label': 'GradientBoostingRegressor', 'value': 'GBR'},
                 {'label': 'LinearRegression', 'value': 'LR'},
                 {'label': 'Lasso', 'value': 'Lasso'},
                 {'label': 'KNeighborsRegressor', 'value': 'KNR'},
                 {'label': 'ElasticNet', 'value': 'EN'},
                 {'label': 'DecisionTreeRegressor', 'value': 'DTR'},
                 {'label': 'Sequential', 'value': 'Sequential'}],
               placeholder=" Please do select a model ")]),
    html.Div(id="output-model"),

    html.Button('Run', id='button'),
    html.Div(id="RUN"),

    dcc.Graph(id='first_graph',
              figure=fig)

])

#from here onwards you have to define the modification done by the user

@app.callback(
    Output(component_id='output-date', component_property='children'),
    Input(component_id='input-date', component_property='date'))

def update_output_date(date_value):
    if date_value is not None:
        choosen_date = date_value
        return ['You have selected "{}"'.format(date_value),choosen_date]

@app.callback(
    Output('output-days', 'children'),
    Input('my-input', 'value'))

def update_output(value):
    prediction_days = value
    return ['You have selected "{}"'.format(prediction_days),prediction_days]

@app.callback(
    Output('output-model', 'children'),
    Input('dropdown_model', 'value'))

def update_model(model):
    choosen_model = model
    return ['You have selected "{}"'.format(choosen_model),choosen_model]

@app.callback(
    Output(component_id='RUN', component_property='children'),
    Input('button', 'n_clicks'),
    State('output-days','children'),
    State('output-model','children'),
    State('output_date','children')
)

def update_output_div():
    return subprocess.call(['sh', './Shield.sh'])

#from here onwards you have to define on which server you want to define your dash

server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
