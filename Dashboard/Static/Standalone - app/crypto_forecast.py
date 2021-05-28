import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from datetime import datetime
import yfinance as yf
from fbprophet import Prophet
import plotly.graph_objects as go

app = dash.Dash()

options=[
            {'label': 'Bitcoin', 'value': 'BTC-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Matic', 'value': 'MATIC-USD'},
            {'label': 'Binance ', 'value': 'BNB-USD'},
            {'label': 'Moon', 'value': 'MOON.SW'},
            {'label': 'Etheureum', 'value': 'ETH-USD'},
            {'label': 'Litecoin', 'value': 'LTC-USD'}
        ]

app.layout = html.Div( children=[
    html.H1(
        children='Stock Forecast Dashboard',
    ),

    html.Div(children='Crypto price forecasting'),
    html.H3('Enter a stock symbol:'),
    dcc.Dropdown(
        id='dropdown_symbol',
        options=options,
        value='BTC-USD',
        multi=False
    ),
    # Submit Button
    html.Div([
        html.Button(id='submit_button',
                    n_clicks=0,
                    children='Submit'
                    )

    ], style={'display': 'inline-block'}),
    dcc.Graph(
        id='graph_scatter'
    )
])

@app.callback(
    Output('graph_scatter', 'figure'),
    [Input('dropdown_symbol', 'value'),
     Input('submit_button', 'n_clicks')])
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
