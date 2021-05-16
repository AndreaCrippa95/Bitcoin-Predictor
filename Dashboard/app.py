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

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/data/DF_train.csv'
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

#from here onwards you have the text, will defined later on.
DatePicker_text = ''' ### Please select the date'''

Dropdown_text = ''' ### MODEL SELECTION '''

#from here onwards you have to define the layout of the Dashb
app.layout = html.Div([
    html.H1(children=' Welcome to our BTC Forecasting Platform.'),
    html.Div(children='Forecast'),
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
    dcc.DatePickerSingle(
        id='my-date-picker-single',
        min_date_allowed=date(2011, 8, 5),
        max_date_allowed=date(2020, 9, 19),
        initial_visible_month=date(2017, 8, 5),
        date=date(2017, 8, 25)
    ),

    html.Button('50 DAYS', id='btn-1'),
    html.Button('100 DAYS', id='btn-2'),
    html.Button('500 DAYS', id='btn-3'),
    html.Div(id='container'),


    dcc.Markdown(children=Dropdown_text),
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
    html.Div(id='output-container-date-picker-single'),

    html.Div(id="my_output")

])

#from here onwards you have to define the modification done by the user

@app.callback(
    [Output(component_id='overview_graph', component_property='figure')],
    [Input(component_id='yaxis-type', component_property='value')])

def update_graph(yaxis_type):
    fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')

    return fig

@app.callback(
    [Output('output-container-date-picker-single', 'children')],
    [Input('my-date-picker-single', 'date')])
def update_output(date_value):
    string_prefix = 'You have selected: '

    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%B %d, %Y')
        return string_prefix + date_string
"""
@app.callback(
    [Output('container', 'children')],
    [Input('btn-1', 'days')],
    [Input('btn-2', 'days')],
    [Input('btn-3', 'days')])
def change_text(days):
    return ["Number of days is " + str(days)]
"""

#from here onwards you have to define on which server you want to define your dash
server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
