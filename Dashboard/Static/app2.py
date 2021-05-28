import os
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

import graphs as gr
import texts as tx

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'text':"#7FDBFF",
    'left':"#E86361",
    'right':"#3A82B5"
}
####### They are ordered by actual Marketcap on the 28.05.2021,
####### I have added the ETP from 21shares called MOON
####### This can work with any ticker from yahoo finance.
options=[
            {'label': 'Bitcoin', 'value': 'BTC-USD'},
            {'label': 'Ethereum', 'value': 'ETH-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Tether ', 'value': 'USDT-USD'},
            {'label': 'Binance ', 'value': 'BNB-USD'},
            {'label': 'Cardano', 'value': 'ADA-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Dogecoin', 'value': 'DOGE-USD'},
            {'label': 'Polygon', 'value': 'MATIC-USD'},
            {'label': 'Moon', 'value': 'MOON.SW'},
            {'label': 'Litecoin', 'value': 'LTC-USD'}
        ]

##### DF & GRAPH
file_path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(file_path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

prediction_days = 100
Real_Price = df.loc[:,'BTC_Price']
fig = go.Figure()
fig.add_trace(go.Line(x=df["Date"], y=df["BTC_Price"],
                    mode='lines',
                    name=df.columns.values[1]))
fig.add_trace(go.Line(x=df["Date"], y=df["Gold_Price"],
                    mode='lines',
                    name=df.columns.values[2]))
fig.add_trace(go.Line(x=df["Date"], y=df["NDAQ_Price"],
                    mode='lines',
                    name=df.columns.values[3]))
fig.layout.plot_bgcolor = '#fff'


##### LAYOUT OF APP
app.layout = html.Div(children=[
        html.H1(children='Welcome to our Dashboard',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    html.H3('Select your crypto form yahoo-finance:'),
    dcc.Dropdown(
        id='dropdown_symbol',
        options=options,
        value='BTC-USD',
        multi=False
    ),
    html.Div([html.H3('Enter start - end date:'),
              dcc.DatePickerRange(
                  id='date_range',
                  #start_date_placeholder_text='Start Date',
                  min_date_allowed=datetime(2015, 1, 1),
                  max_date_allowed=datetime.now(),
                  start_date=datetime(2019, 1, 1),
                  end_date=datetime.now(),
                  number_of_months_shown=2
    )

    ], style={'display': 'inline-block'}),
    html.Div([
        html.Button(id='submit_button',
                    n_clicks=0,
                    children='Submit',
                    style={'fontSize': 18, 'marginLeft': '30px'}

                    )
    ], style={'display': 'inline-block'}),
    dcc.Graph(
        id='graph_scatter'
    ),
    dcc.Graph(
        id='graph_candle'
    ),
    dcc.Graph(
        id='graph_volume'
    ),

    html.H2(children='Actual Data and Graphs.',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    dcc.Graph(
            style={'width': '500', 'backgroundColor': 'white'},
            id='overview_graph',
            figure=fig),

    html.Br(),
    html.H2(id='text_results', children= 'The Results parts',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.BM_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

   dcc.Markdown(
        tx.BM_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.H1(
        children='Stock Forecast Dashboard',
    ),

    html.Div(children='Crypto price forecasting'),
    html.H3('Enter a stock symbol:'),
    dcc.Dropdown(
        id='dropdown_symbol_2',
        options=options,
        value='BTC-USD',
        multi=False
    ),
    # Submit Button
    html.Div([
        html.Button(id='submit_button_2',
                    n_clicks=0,
                    children='Submit'
                    )

    ], style={'display': 'inline-block'}),
    dcc.Graph(
        id='graph_scatter_2'
    )


])

############FIRST PART OF OUR DASHBOARD
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

    figure = {'data': data,
              'layout': {
                  'title': str(symbol) + " closing value"}
              }
    return figure

@app.callback(
    Output('graph_candle', 'figure'),
    [Input('dropdown_symbol', 'value'),
     Input('date_range', 'start_date'),
     Input('date_range', 'end_date'),
     Input('submit_button', 'n_clicks')])
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

    figure = {'data': data,
              'layout': {
                  'title': str(symbol) + " candlestick chart"}
              }
    return figure

@app.callback(
    Output('graph_volume', 'figure'),
    [Input('dropdown_symbol', 'value'),
     Input('date_range', 'start_date'),
     Input('date_range', 'end_date'),
     Input('submit_button', 'n_clicks')])
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

    figure = {'data': data,
              'layout': {
                  'title': str(symbol) + " trading volume",
                 'size': 18
                  }}

    return figure

#####DISPLAY OF ACTUAL RESULTS BM
@app.callback(
    [Output(component_id='text-display',component_property='children')],
    [Input(component_id='text-input',component_property='value')])

def update_text_output_2(input_value):
    with open(path1, 'r') as Texlist1:
        content = Texlist1.read()
    return content

#####PREDICTIONS VIA FBPROPHET

@app.callback(
    Output('graph_scatter_2', 'figure'),
    [Input('dropdown_symbol_2', 'value'),
     Input('submit_button_2', 'n_clicks')])
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
        name="Data values"
    )

    yhat = go.Scatter(
        x=forecast1.index,
        y=forecast1["yhat"],
        mode='lines',
        name="Forecast"
    )

    yhat_upper = go.Scatter(
        x=forecast1.index,
        y=forecast1["yhat_upper"],
        mode='lines',
        fill="tonexty",
        name="Higher uncertainty interval"
    )

    yhat_lower = go.Scatter(
        x=forecast1.index,
        y=forecast1["yhat_lower"],
        mode='lines',
        fill="tonexty",
        name="Lower uncertainty interval"
    )

    data = [historic, yhat, yhat_upper, yhat_lower]

    figure = {'data': data,
              'layout': {
                  'title': str(symbol) + " closing value"}
              }

    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
