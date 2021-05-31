import os
import sys
import pandas as pd
from datetime import datetime
import sqlite3
import base64
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from pandas_datareader import data as web
import yfinance as yf
from fbprophet import Prophet

import support.graphs as gr
import support.texts as tx
import random
import time

from DataClass import Data
from MethodClass import Method
from ResultClass import Results

####################################################################################################
# 001 - JS & CSS, Layout
####################################################################################################

# Java
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# CSS
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

# Chosen colors
colors = {
    'text':"#7FDBFF",
    'left':"#E86361",
    'right':"#3A82B5"
}
####################################################################################################
# 002 - Necessary references and options
####################################################################################################

options=[
            {'label': 'Bitcoin', 'value': 'BTC-USD'},
            {'label': 'Ethereum', 'value': 'ETH-USD'},
            {'label': 'Tether ', 'value': 'USDT-USD'},
            {'label': 'Binance ', 'value': 'BNB-USD'},
            {'label': 'Cardano', 'value': 'ADA-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Dogecoin', 'value': 'DOGE-USD'},
            {'label': 'USD Coin', 'value': 'USDC-USD'},
            {'label': 'Polkadot', 'value': 'DOT-USD'},
            {'label': 'Polygon', 'value': 'MATIC-USD'},
            {'label': 'Moon ETP', 'value': 'MOON.SW'}
        ]

list_of_ourgraphs = [
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_BM.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_DNN.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_DTR.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_EN.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_GBR.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_KNR.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_Lasso.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_LR.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_RFR.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_Sequential.png',
        '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/Graphs/Bitcoin_SVM.png'
        ]

list_of_our_methods = [
    'Brownian Motion',
    'Linear Regression',
    'Lasso',
    'Elastic Net',
    'K-neighbour Regressor',
    'Decision Tree Regressor',
    'Random Forest Regressor',
    'Gradient Boosting Regressor',
    'Support Vector Machine Regressor',
    'Sequential Neural Network',
    'Dense Neural Network'
]

list_of_our_days = [5,10,15,20]

list_of_our_variables = ['BTC Price', 'Gold Price', 'NASDAQ Price','BTC Returns']

# DF Graph
file_path = '/Users/andreacrippa/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(file_path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

prediction_days = 100
Real_Price = df.loc[:,'BTC_Price']
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"],
                            y=df["BTC_Price"],
                                mode='lines',
                                    name=df.columns.values[1],
                                        line=dict(color='rgb(5, 108, 159)')))
fig.add_trace(go.Scatter(x=df["Date"],
                            y=df["Gold_Price"],
                                mode='lines',
                                    name=df.columns.values[2],
                                        line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df["Date"],
                            y=df["NDAQ_Price"],
                                mode='lines',
                                    name=df.columns.values[3],
                                        line=dict(color='red')))
fig.layout.plot_bgcolor = '#fff'
fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)',
                    plot_bgcolor = 'rgba(0,0,0,0)',
                        xaxis = {'showgrid': True},
                            yaxis = {'showgrid': True},
                                yaxis_zeroline=True,
                                    xaxis_zeroline=True)


####################################################################################################
# 003 - APP Start
####################################################################################################

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                    external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[

        html.H1(children='WELCOME TO OUR DASHBOARD',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                ),
    html.Br(),

        html.H3(children='Project designed by Andrea CRIPPA, Maximilian SETZER , Flavio ROJO.'
                         ,style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                                 }
                ),

    html.Br(),

        dcc.Markdown('COPY THE ABSTRACT HERE'
                     ,style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                             }
                     ),

    html.Br(),

        html.H2(children='CHARTS SECTION',
                style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                       }
                ),

    html.Br(),

        html.H5('Select your cryptocurrency, from Yahoo Finance'),

        dcc.Dropdown(id='dropdown_symbol',
                        options=options,
                            value='BTC-USD',
                                multi=False
                    ),

        html.Div(
                  dcc.DatePickerRange(id='date_range',
                                        min_date_allowed=datetime(2015, 1, 1),
                                            max_date_allowed=datetime.now(),
                                                start_date=datetime(2015, 1, 1),
                                                    end_date=datetime.now(),
                                                        number_of_months_shown=2
                                      ),

                        style={'display': 'inline-block','font-family': 'Helvetica'
                              }
                ),
        html.Div([
            html.Button(id='submit_button',
                            n_clicks=0,
                                children='Submit',
                                     style={'fontSize': 10, 'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                                            }
                        )
                ],
                        style={'display': 'inline-block'
                               }
                ),

        html.H3(children='CLOSING VALUE GRAPH',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                ),

        dcc.Graph(id='graph_scatter'
                ),

        html.H3(children='CANDLESTICK GRAPH',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                        }
                ),

        dcc.Graph(id='graph_candle'
                ),

        html.H3(children='TRADING VOLUME GRAPH',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                       }
                ),

        dcc.Graph(id='graph_volume'
                 ),

        html.H3(children='COMPARABLE GRAPH',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                ),

        dcc.Graph(id='overview_graph',
                     figure=fig
                  ),

    html.Br(),

        html.H2(children='ADA PROJECT RESULTS SECTION',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                ),

    html.Br(),
        dcc.Markdown('COPY THE ABSTRACT HERE',
                     style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                            }
                     ),
    html.Br(),

    html.Br(),
        dcc.Dropdown(id='graph_dropdown_BTC_price',
                    options=[{'label': i, 'value': i} for i in list_of_our_methods],
                        value=list_of_our_methods[0],
                            placeholder="Select a method",

                ),
        dcc.Dropdown(id='graph_dropdown_BTC_choice',
                    options=[{'label': i, 'value': i} for i in list_of_our_variables],
                        value=list_of_our_variables[0],
                            placeholder="Select some variables",
                                multi=True

                ),

        html.Div(
                          dcc.DatePickerRange(id='date_range_ADA',
                                                min_date_allowed=datetime(2015, 1, 1),
                                                    max_date_allowed=datetime.now(),
                                                        start_date=datetime(2019, 1, 1),
                                                            end_date=datetime.now(),
                                                                number_of_months_shown=2
                                              ),

                                style={'display': 'inline-block','font-family': 'Helvetica'
                                      }
                        ),

        dcc.Dropdown(id='dropdown_days',
                         options=[
                             {'label': '5 days', 'value': 5},
                             {'label': '10 days', 'value': 10},
                             {'label': '15 days', 'value': 15},
                             {'label': '20 days', 'value': 20}
                         ],
                                    value=5,
                                        multi=False
                            ),

        html.Div([
                    html.Button(id='submit_button_ADA',
                                    n_clicks=0,
                                        children='Submit',
                                             style={'fontSize': 10, 'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                                                    }
                                )
                        ],
                                style={'display': 'inline-block'
                                       }
                        ),

        html.Img(id='BTC_image',
                    alt='The image can not be displayed try later.',
                        style={'text-align': 'center', 'display': 'inline-block', 'width': '100%',
                            'max-width': '800px','vertical-align': 'middle', 'horizontal-align':'middle'}
                 ),

    html.Br(),

    html.Br(),

        html.H2(children='CRYPTO FORECAST VIA FBPROPHET',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                ),

    html.Br(),

            html.H5('Select your cryptocurrency, from Yahoo Finance'),
                dcc.Dropdown(id='dropdown_symbol_2',
                        options=options,
                            value='BTC-USD',
                                multi=False,

            ),

        html.Div([
            html.Button(id='submit_button_2',
                            n_clicks=0,
                                style={'display': 'inline-block','font-family': 'Helvetica'},
                                    children='Submit'
                        )
                 ],
                    style={'display': 'inline-block'}),

        dcc.Graph(id='graph_scatter_2'
                ),

    html.Br(),

        html.H2(children='TWITTER SENTIMENT',
                    style={'textAlign': 'center','color': colors,'font-family': 'Helvetica'
                           }
                        ),

    html.Br(),

        html.Div(children = [
                        dcc.Input(id='sentiment_term', value='Bitcoin', type='text'),
                            html.Div(id='input-div', style={'color': 'rgb(255,255,255)'}
                        )
                    ]
                ),
        html.Div(children = [
                        html.Button('Submit', id="submit-button"
                        )
                    ]
                ),
        html.Div(children = [
                        html.Div(children = [
                            dcc.Graph(id='live-graph', animate=False
                        )
                    ]
                ),
            ]
        ),

        dcc.Interval(id='graph-update',
                       interval=1*1000
                    )
                ]
            )

####################################################################################################
# 004 - Callbacks
####################################################################################################

# Closing-price Graph
@app.callback(
    Output(component_id='graph_scatter', component_property='figure'),
    [Input(component_id='dropdown_symbol', component_property='value'),
     Input(component_id='date_range', component_property='start_date'),
     Input(component_id='date_range', component_property='end_date'),
     Input(component_id='submit_button', component_property='n_clicks')])
def update_scatter(symbol, start_date, end_date, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    else:
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=start_date, end=end_date)

    first = go.Scatter(x=df.index,
                       y=df['Close'])

    data = [first]

    figure = {'data': data}
    return figure

# candlestick Graph
@app.callback(
    Output(component_id='graph_candle', component_property='figure'),
    [Input(component_id='dropdown_symbol', component_property='value'),
     Input(component_id='date_range', component_property='start_date'),
     Input(component_id='date_range', component_property='end_date'),
     Input(component_id='submit_button', component_property='n_clicks')])
def update_graph(symbol, start_date, end_date, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    else:
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=start_date, end=end_date)

    first = go.Candlestick(x=df.index,
                            open=df['Open'],
                                high=df['High'],
                                    low=df['Low'],
                                        close=df['Close'])

    data = [first]

    figure = {'data': data}
    return figure

# Volume Graph
@app.callback(
    Output(component_id='graph_volume', component_property='figure'),
    [Input(component_id='dropdown_symbol', component_property='value'),
     Input(component_id='date_range', component_property='start_date'),
     Input(component_id='date_range', component_property='end_date'),
     Input(component_id='submit_button', component_property='n_clicks')])
def update_scatter(symbol, start_date, end_date, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    else:
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=start_date, end=end_date)

    first = go.Scatter(x=df.index,
                       y=df['Volume'])

    data = [first]

    figure = {'data': data}

    return figure


# Results Dropdown
@app.callback(
    Output(component_id='BTC_image', component_property='src'),
    [Input(component_id='graph_dropdown_BTC_price',component_property= 'value'),
     Input(component_id='graph_dropdown_BTC_choice',component_property='value'),
     Input(component_id='date_range_ADA', component_property='start_date'),
     Input(component_id='date_range_ADA', component_property='end_date'),
     Input(component_id='dropdown_days', component_property='value'),
     Input(component_id='submit_button_ADA', component_property='n_clicks')])
def update_image_src(method,variables,start_date,end_date,pred_days,n_clicks):
    if 'BTC Price' in variables:
        BTC = True
    else:
        BTC = False

    if 'Gold Price' in variables:
        Gold = True
    else:
        Gold = False

    if 'NASDAQ Price' in variables:
        NDAQ = True
    else:
        NDAQ = False

    if 'BTC Returns' in variables:
        Returns = True
    else:
        Returns = False

    if method == 'Brownian Motion':
        image_path = list_of_ourgraphs[0]
        ChModel = 'BR'
    elif method == 'Linear Regression':
        image_path = list_of_ourgraphs[7]
        ChModel = 'LR'
    elif method == 'Lasso':
        image_path = list_of_ourgraphs[6]
        ChModel = 'Lasso'
    elif method == 'Elastic Net':
        image_path = list_of_ourgraphs[3]
        ChModel = 'EN'
    elif method == 'K-neighbour Regressor':
        image_path = list_of_ourgraphs[5]
        ChModel = 'KNR'
    elif method == 'Decision Tree Regressor':
        image_path = list_of_ourgraphs[2]
        ChModel = 'DTR'
    elif method == 'Random Forest Regressor':
        image_path = list_of_ourgraphs[8]
        ChModel = 'RFR'
    elif method == 'Gradient Boosting Regressor':
        image_path = list_of_ourgraphs[4]
        ChModel = 'GBR'
    elif method == 'Support Vector Machine Regressor':
        image_path = list_of_ourgraphs[10]
        ChModel = 'SVR'
    elif method == 'Sequential Neural Network':
        image_path = list_of_ourgraphs[9]
        ChModel = 'Sequential'
    elif method == 'Dense Neural Network':
        image_path = list_of_ourgraphs[1]
        ChModel = 'DNN'
    else:
        image_path = list_of_ourgraphs[0]
        ChModel = 'BR'

    prediction_days = int(pred_days)

    if n_clicks == 0:
        print(start_date,end_date)
    else:
        dat = Data(start=start_date, end=end_date, days=prediction_days, BTC=BTC, Gold=Gold, NDAQ=NDAQ,
                   Returns=Returns)
        dat.create_data()
        df = dat.df
        met = Method(df, ChModel=ChModel, days=prediction_days, Data=dat)
        if ChModel == 'BM':
            res = met.Brownian_Motion()
        elif ChModel == 'Sequential':
            res = met.Sequential()
        elif ChModel in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
            res = met.MachineLearning()
        elif ChModel in ['SVM']:
            res = met.SVM()
        elif ChModel in ['DNN']:
            res = met.DNN()
        else:
            raise ValueError

        gmaker = Results(df, res, ChModel=ChModel, end=end_date, days=prediction_days)
        gmaker.Graph()

    time.sleep(10)
    print('current image_path = {}'.format(image_path))
    encoded_image = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


# Prediction via FBProphet
@app.callback(
    Output(component_id='graph_scatter_2', component_property='figure'),
    [Input(component_id='dropdown_symbol_2', component_property='value'),
     Input(component_id='submit_button_2', component_property='n_clicks')])
def update_scatter(symbol, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    else:
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    prophet_df = df.copy()
    prophet_df.reset_index(inplace=True)
    prophet_df = prophet_df.rename(columns={"Date": "ds", "Close": "y"})

    forcast_columns = ['yhat', 'yhat_lower', 'yhat_upper']
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=100)
    forecast = m.predict(future)
    forecast = forecast.rename(columns={"ds": "Date"})
    forecast1 = forecast.set_index("Date")
    forecast1 = forecast1[datetime.now():]

    historic = go.Scatter(
        x=df.index,
            y=df["Close"],
                name="Historic value"
    )

    yhat = go.Scatter(
        x=forecast1.index,
            y=forecast1["yhat"],
            mode='lines',
                line=dict(color='rgb(230,0,0)'),
                    name="Forecast line"
    )

    yhat_upper = go.Scatter(
        x=forecast1.index,
            y=forecast1["yhat_upper"],
                mode='lines',
                    fill="tonexty",
                        line=dict(color='rgb(128,170,255)'),
                            name="High uncertainty"
    )

    yhat_lower = go.Scatter(
        x=forecast1.index,
            y=forecast1["yhat_lower"],
                mode='lines',
                    fill="tonexty",
                        line=dict(color='rgb(0,170,204)'),
                            name="Low uncertainty"
    )

    data = [historic, yhat, yhat_upper, yhat_lower]

    figure = {'data': data}

    return figure

# Twitter sentiment
@app.callback(Output(component_id='input-div', component_property='children'),
              [Input(component_id='submit-button', component_property='n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])
def update_div(n_clicks, input_value):
    return input_value

@app.callback(Output(component_id='live-graph', component_property='figure'),
              [Input(component_id='graph-update', component_property='interval'),
               Input(component_id='input-div', component_property='children')])

def update_graph_scatter(n, input_value):
    try:
        conn = sqlite3.connect('Twitter/twitter.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 1000", conn ,params=('%' + input_value + '%',))
        df.sort_values('unix', inplace=True)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()

        df['date'] = pd.to_datetime(df['unix'],unit='ms')
        df.set_index('date', inplace=True)

        df = df.resample('100ms').mean()
        df.dropna(inplace=True)

        X = df.index
        Y = df.sentiment_smoothed

        data = go.Scatter(x=X,
                             y=Y,
                                name='Scatter',
                                    mode= 'lines+markers'
                        )

        return {'data': [data],'layout' : go.Layout(
                                                xaxis=dict(range=[min(X),max(X)]),
                                                        yaxis=dict(range=[min(Y),max(Y)]))}

    except:
        pass

####################################################################################################
# 005 - Server displays
####################################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
