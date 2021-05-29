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

# Dataset
file_path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(file_path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

prediction_days = 100
Real_Price = df.loc[:,'BTC_Price']

#Descriptions
path1 = os.path.join('/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/Descriptions/BM.txt')


# Options via yahoo finance ticker

options=[
            {'label': 'Bitcoin', 'value': 'BTC-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Matic', 'value': 'MATIC-USD'},
            {'label': 'Binance ', 'value': 'BNB-USD'},
            {'label': 'Moon', 'value': 'MOON.SW'},
            {'label': 'Etheureum', 'value': 'ETH-USD'},
            {'label': 'Litecoin', 'value': 'LTC-USD'}
        ]

#Graphs
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

app.layout = html.Div(style={'backgroundColor': 'white', 'color': 'black'},
                    children=[

    html.H1(children='Welcome to our Dashboard',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    html.H5(children='To refresh the graphs and predictions, please do refer to our Readme.md file',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    dcc.Dropdown(
        id='my-dropdown',
        options=options,
        value='BTC-USD'
    ),
    dcc.Graph(id='my-graph', style={'width': '500'}),


    dcc.Markdown(tx.First_desc_markdown,style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    html.Br(),
    html.H2(children='Actual Data and Graphs.',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    html.H5(children='This is the graphic representation of our DataFrame, '
                     'which comprises the data collected since 2013 till today, '
                     'our dataframe has three main columns, the BTC_Price'
                     'the Gold_Price and finally the NASDAQ Price.'
                    ,style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

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

    html.Img(
        src='data:image/png;base64,{}'.format(gr.EN_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.EN_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.GBR_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.GBR_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.KNR_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.KNR_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.Lasso_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.Lasso_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.LR_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.LR_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.RFR_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.RFR_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.Seq_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    dcc.Markdown(
        tx.Sequential_txt_markdown,
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Img(
        src='data:image/png;base64,{}'.format(gr.SVm_png64),
        height=300,
        style={'display': 'inline-block', 'vertical-align': 'middle'},
        alt='The image can not be displayed try later.'),

    html.Br(),
    html.H2(children='Statistical analysis of our models',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    dcc.Markdown(
        '''*This text will be in italic*''',
        style={'display': 'inline-block', 'vertical-align': 'middle'}),

    html.Div(children=[
        html.H2(children='Crypto price forecasting chart web application framework for Python.'),
        html.H3('Enter a stock symbol:'),

            dcc.Dropdown(
                id='dropdown_symbol',
                options=options,
                value='BTC-USD',
                multi=False
            ),

            html.Div([
                html.Button(id='submit_button',
                            n_clicks=0,
                            children='Submit'),
                dcc.Graph(
                    id='graph_scatter')
                    ])
]),

    html.H2(children='Live Twitter Sentiment', style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    html.Div([
    html.Div(
        className = 'container-fluid',
        children =[html.H2('Live Twitter Sentiment', className = 'header-title')],
        ),
    html.Div(
        className = 'row search',
        children = [
            html.Div(
                className = 'col-md-4 mb-4',
                children = [html.H5('SearchTerm :', className = 'keyword')]
                     ),
            html.Div(
                className = 'col-md-4 mb-4',
                children = [
                    dcc.Input(id='sentiment_term', className = 'form-control', value='Twitter', type='text'),
                    html.Div(['example'], id='input-div', style={'display': 'none'}),
                    ]
                ),
            html.Div(
                className = 'col-md-4 mb-4',
                children = [
                    html.Button('Submit', id="submit-button" ,className = 'btn btn-success'),
                    ]
                ),
            ]
        ),
    html.Div(
        className = 'row',
        children = [
            html.Div(
                className = 'col-md-8 mb-8',
                children = [
                    dcc.Graph(id='live-graph', animate=False),
                    ]
                ),
            html.Div(
                className = 'col-md-4 mb-4',
                children = [
                    dcc.Graph(id='sentiment-pie', animate=False),
                    ]
                ),
            ]
        ),

      dcc.Interval(id='graph-update',
                   interval=1*1000
                ),
            ]
     )
])

# For the Graphs

@app.callback(Output('my-graph', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(selected_dropdown_value,'yahoo',datetime(2015, 1, 1),datetime.now())
    return {
        'data': [{
            'x': df.index,
            'y': df.Close
        }],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
    }

'''
For the Descriptions
@app.callback(
    [Output(component_id='text-display',component_property='children')],
    [Input(component_id='text-input',component_property='value')])


def update_text_output_2(input_value):
    with open(path1, 'r') as Texlist1:
        content = Texlist1.read()
    return content
'''

### Prediction part

@app.callback(
    Output('graph_scatter', 'figure'),
    [Input('dropdown_symbol', 'value'),
     Input('submit_button', 'n_clicks')])
def update_scatter(symbol, n_clicks):
    if n_clicks == 0:
        ticker_data = yf.Ticker('BTC-USD')
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    else:
        #columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        ticker_data = yf.Ticker(symbol)
        df = ticker_data.history(period='1d', start=datetime(2015, 1, 1), end=datetime.now())

    prophet_df = df.copy()
    prophet_df.reset_index(inplace=True)
    # Renaming needed for fbprophet to make predictions
    prophet_df = prophet_df.rename(columns={"Date": "ds", "Close": "y"})

    # Call Prophet() to make prediction up to 100 days
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
        line={"color": "#57b8ff"},
        name="Higher uncertainty interval"
    )

    yhat_lower = go.Scatter(
        x=forecast1.index,
        y=forecast1["yhat_lower"],
        mode='lines',
        fill="tonexty",
        line={"color": "#57b8ff"},
        name="Lower uncertainty interval"
    )

    data = [historic, yhat, yhat_upper, yhat_lower]

    figure = {'data': data,
              'layout': {
                  'title': str(symbol) + " closing value"}
              }

    return figure

# Twitter Feed, works with SQL DB that you have to run in advance. --> please, do refer to our Readme.md

@app.callback(Output(component_id='input-div', component_property='children'),
              [Input(component_id='submit-button', component_property='n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])

def update_div(n_clicks, input_value):
    return input_value

@app.callback(Output(component_id='live-graph',component_property= 'figure'),
              [Input(component_id='graph-update', component_property= 'interval'),
               Input(component_id='input-div', component_property='children')])

def update_graph_scatter(n, input_value):
    try:
        conn = sqlite3.connect('twitter.db')
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

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),
                                                    title='{}'.format(input_value))}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')


server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')