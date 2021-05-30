import os
import sys
import pandas as pd
from datetime import datetime
import sqlite3

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

import os
import sys
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor'
sys.path.append(path)

#Import Data making class, Method caller class and Result maker class
from DataClass import Data
from MethodClass import Method
from ResultClass import Results
#Inputs file. Save globally the variables to send to the various files
import datetime as dt

options=[
            {'label': 'BM', 'value': 'BM'},
            {'label': 'Sequential', 'value': 'Sequential'},
            {'label': 'RFR ', 'value': 'RFR'},
            {'label': 'GBR', 'value': 'GBR'},
            {'label': 'LR', 'value': 'LR'},
            {'label': 'Lasso', 'value': 'Lasso'},
            {'label': 'KNR', 'value': 'KNR'},
            {'label': 'EN', 'value': 'EN'},
            {'label': 'SVM', 'value': 'SVM'},
            {'label': 'DNN', 'value': 'DNN'},
            {'label': 'RNN', 'value': 'RNN'}
        ]





app = dash.Dash(__name__)

app.layout = html.Div(children=[

        dcc.Dropdown(id='dropdown_method',
                        options=options,
                            value='BM',
                                multi=False
                    ),

        html.Div(
                  dcc.DatePickerRange(id='date_range',
                                        min_date_allowed=datetime(2012,1,1),
                                            max_date_allowed=datetime(2021,5,24),
                                                start_date=datetime(2019, 1, 1),
                                                    end_date=datetime.now(),
                                                        number_of_months_shown=1
                                      ))

    ]
)

@app.callback(
    Output('graph_scatter', 'figure'),
    [Input('dropdown_symbol', 'value'),
     Input('date_range', 'start_date'),
     Input('date_range', 'end_date'),
     Input('submit_button', 'n_clicks')])
def update_scatter(symbol, start_date, end_date, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2017, 1, 1), end=datetime.now())

    else:
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=start_date, end=end_date)

    first = go.Scatter(x=df.index,
                       y=df['Close'])

    data = [first]

    figure = {'data': data}
    return figure



#set start, end times and number of days predicted in the future
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,5,24)
prediction_days = 10
#Choose the input data
BTC_Price = True
Gold_Price = True
NDAQ_Price = True
Returns = False

ChModel = 'LR'

RES = True
GRA = True
ACC = True

dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
dat.create_data()
df = dat.df
met = Method(df,ChModel=ChModel,days=prediction_days,Data=dat)
if ChModel == 'BM':
    res = met.Brownian_Motion()
elif ChModel =='Sequential':
    res = met.Sequential()
elif ChModel in ['RFR', 'GBR', 'LR','Lasso','KNR','EN','DTR']:
    res = met.MachineLearning()
elif ChModel in ['SVM']:
    res = met.SVM()
elif ChModel in ['DNN']:
    res = met.DNN()
elif ChModel in ['RNN']:
    res = met.RNN()
else:
    raise ValueError

gmaker = Results(df,res,ChModel=ChModel,end=end,days=prediction_days)
if GRA:
    gmaker.Graph()
if RES:
    gmaker.Results()






if __name__ == '__main__':
    app.run_server(debug=True)
