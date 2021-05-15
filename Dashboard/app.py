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

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(path, header=0)
df.head()
df.columns.values[0] = 'Date'
df.columns.values[2] = 'Gold'
df.columns.values[3] = 'NASDAQ'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#From here onwards you have to define all the functions parts of your dash
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

dcc.RadioItems(
    options=[
        {'label': 'BTC PRICE', 'value': 'BTC'},
        {'label': 'GOLD PRICE', 'value': 'GLD'},
        {'label': 'NASDAQ', 'value': 'NAS'}
    ],
    value='BTC')

#from here onwards you have the text
DatePicker_text = '''
### 100 DAYS

From here on you can choose the timeframe to make a prediction
Dash uses the [CommonMark](http://commonmark.org/)
'''

Dropdown_text = '''
### MODEL SELECTION


MODEL 1 : In statistics, linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables).
MODEL 2 : A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence.
MODEL 3 : In mathematics, a random walk is a mathematical object, known as a stochastic or random process, that describes a path that consists of a succession of random steps on some mathematical space such as the integers.

From here on you can choose the model for the predictions
Dash uses the [CommonMark](http://commonmark.org/)
'''

#from here onwards you have to define the layout of the Dashb
app.layout = html.Div([
    html.H1(children=' Welcome to our BTC Forecasting Platform.'),
    html.Div(children='The forecast in this blog are for general informational purposes only and are not intended to provide specific advice financial advise.'),
    html.Div(children='General overview.'),
    dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
        ),
    dcc.Graph(id='overview_graph',
        figure=fig
        ),

    dcc.Markdown(children=DatePicker_text),
    dcc.DatePickerRange(
        start_date=date(2012,1,1),
        end_date=date(2019,1,1),
        minimum_nights=100,
        clearable=True,
        with_portal=True
        ),


    dcc.Markdown(children=Dropdown_text),
    dcc.Dropdown(id="dropdown_model",
        options=[
            {'label': 'Linear Regression', 'value': 'LR'},
            {'label': 'RNN', 'value': 'RNN'},
            {'label': 'Random Walk', 'value': 'RW'}],
        placeholder="Select a model"),

    html.Div(id="my_output")

])

#from here onwards you have to define the modification done by the user

@app.callback(
    Output(component_id='overview_graph', component_property='figure'),
    Input(component_id='yaxis-type', component_property='value'))

def update_graph(yaxis_type):
    fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')

    return fig

#from here onwards you have to define on which server you want to define your dash
server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
