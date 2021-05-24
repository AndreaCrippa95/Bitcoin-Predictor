import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from MethodClass import Method
from DataClass import  Data
from ResultClass import Results

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'

app.layout = html.Div([
    html.H1(id='text_header', children=' Welcome to our BTC Forecasting Platform.'),

    dcc.Markdown(id='text_description', children='Text description of the project'),

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

    html.Div(
        dcc.Store(id="memory")
    ),

    html.Button('Run', id='button'),
    html.Div(id="RUN"),

])

# from here onwards you have to define the modification done by the user

@app.callback(
    Output(component_id='output-date', component_property='children'),
    Input(component_id='input-date', component_property='date'))
def update_output_date(date_value):
    if date_value is not None:
        return date_value


@app.callback(
    [Output('output-days', 'children')],
    [Output('memory', 'data')],
    Input('my-input', 'value'))
def update_output(value):
    return value


@app.callback(
    [Output('output-model', 'children')],
    [Output('memory', 'data')],
    Input('dropdown_model', 'value'))
def update_model(model):
    return model


@app.callback(
    Output(component_id='RUN', component_property='children'),
    [Input('button', 'n_clicks')],
    [Input('memory', 'data')],
    State('output-days', 'children'),
    State('output-model', 'children'),
    State('output_date', 'children')
)
def update_output_div():
    start= date(2012,1,1)
    end = update_output_date()
    prediction_days = update_output()
    ChModel = update_model
    BTC_Price = True
    Gold_Price = False
    NDAQ_Price = False
    GRA = True
    RES = True
    ACC = True
    dat = Data(start=start, end=end, days=prediction_days, BTC=BTC_Price, Gold=Gold_Price, NDAQ=NDAQ_Price)
    df = dat.create_data()
    met = Method(df, ChModel=ChModel, days=prediction_days)
    if ChModel == 'BM':
        res = met.Brownian_Motion()
    elif ChModel == 'Sequential':
        res = met.Sequential()
    elif ChModel in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
        res = met.MachineLearning()
    elif ChModel in ['SVM']:
        res = met.SVM()

    gmaker = Results(df, res, ChModel=ChModel, end=end, days=prediction_days)
    if GRA:
        gmaker.Graph()
    if RES:
        gmaker.Results()
    if ACC:
        gmaker.Accuracy()
    return


# from here onwards you have to define on which server you want to define your dash

server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
