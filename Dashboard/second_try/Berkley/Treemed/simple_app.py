import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio

import os
import sys
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/second_try/Berkley/predictor/datastore'
sys.path.append(path)

from datastore import datastore
from datetime import datetime, date, time, timedelta
import time
import ta

d = datastore.DataStore(None, realtime_prices=True)
time.sleep(60 - datetime.now().second)


def countX(lst, x):
    """
    Counts the number of times 'x' appeared in the list lst
    """
    return lst.count(x)

def fill_trends(df):
    """
    Computes the various indicators using the closing price and updates the dataframe
    with new columns of these indicators

    Returns:
    --------
    Dataframe with indicators added as new columns
    """
    # exponential moving averages of various timeperiods
    df['ema_5'] = ta.trend.EMAIndicator(close = df['close'], n = 5).ema_indicator()
    df['ema_13'] = ta.trend.EMAIndicator(close = df['close'], n = 13).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(close = df['close'], n = 21).ema_indicator()
    df['ema_34'] = ta.trend.EMAIndicator(close = df['close'], n = 34).ema_indicator()
    df['ema_55'] = ta.trend.EMAIndicator(close = df['close'], n = 55).ema_indicator()
    df['ema_100'] = ta.trend.EMAIndicator(close = df['close'], n = 100).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close = df['close'], n = 200).ema_indicator()

    # ichimoku cloud
    ichi_ind = ta.trend.IchimokuIndicator(high = df['high'], low = df['low'],n1=9, n2=26, n3=52)
    df['ichi_lead_a'] = ichi_ind.ichimoku_a()
    df['ichi_lead_b'] = ichi_ind.ichimoku_b()
    df['ichi_base'] = ichi_ind.ichimoku_base_line()
    df['ichi_conver'] = ichi_ind.ichimoku_conversion_line()
    df['ichi_lag'] = df['close'].shift(-26)

    return df.fillna(0.0)

data = predictor.history()
data

# This will plot predictions and actual prices on the same graph
import plotly.express as px
plot = px.line(data, x="time", y=['actual', 'prediction'])
plot.show()

#Build App
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
colors = {"background": 'rgb(46, 60, 88)','text': '#fafafa'}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Dashboard Design

#Row 1: BTC Graph
row_1 = [html.H2("Bitcoin Price Forecasts"),
         dcc.Graph(id='btc-graph'),
         dcc.Interval(id = 'btc-update',
                      interval = 30 * 1000, #Update graph every 30 seconds (in milliseconds)
                      n_intervals = 0,
                      max_intervals = -1
                    )
]


# Setting layout for the application
app.layout = html.Div([body])

#Real-Time Results: BTC Graph
@app.callback(
    Output('btc-graph', 'figure'),
    [Input('btc-update', 'n_intervals')]
)
def btc_updates(n_intervals):
    if (predictor.has_model()):
        now = datetime.now()
        print("BTC Graph Updated At:" + str(now))

        #Prediction history + new predictiom
        df = predictor.history(100)
        new_time = df['time'].iloc[-1] + timedelta(seconds=60)
        new_prediction = predictor.predict()
        new_row = {'time':new_time, 'actual':None, 'prediction':new_prediction}
        new_df = df.append(new_row, ignore_index=True)


        #Plot
        data1 = go.Scatter(
            x=new_df['time'], y=new_df['actual'], name='actual',line=dict(color='blue', width=4)
        )
        data2 = go.Scatter(
            x=new_df['time'], y=new_df['prediction'], name='prediction', line=dict(color='red', width=4)
        )
        layout_btc = {
            'font': {
                'family':'Anaheim',
                'size':30,
                'color': colors['background']
            }
        }
        myfig = {'data' : [data1,data2],
                'layout': layout_btc}
        return myfig

app.run_server(debug=True)
