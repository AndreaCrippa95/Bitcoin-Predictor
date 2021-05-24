import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import dash_html_components as html
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from numpy import radians, cos, sin


from datastore import datastore
from datetime import datetime, date, time, timedelta
import time
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.io as pio
import plotly.graph_objects as go


# In[6]:


# Keep `realtime_prices` on so there's no missing data
d = datastore.DataStore(None, realtime_prices=True)

# The cache needs a bit of time to warm up
time.sleep(60 - datetime.now().second)


# ## Technical Analysis - Exponential Moving Average (EMA) & Ichimoku Cloud

# In[7]:


def countX(lst, x):
    """
    Counts the number of times 'x' appeared in the list lst
    """
    return lst.count(x)


# In[8]:


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


# ## Generate Trading Signals Using EMA Crossover Strategy

# In[9]:


def ma_crossover(df, close, slow, fast):
    """
    Computes the position to take at each point in time
    based on the moving average crossover strategy for various range of time periods

    If the slower moving average rises above the faster moving average, signal is 1.
    Else is 0

    The position take into consideration the signal given.
    Position: Buy if 1 or Sell if -1

    Returns dataframe df with signal and position
    """
    df['close'] = close
    df['slow'] = slow
    df['fast'] = fast
    df['signal'] = 0.0
    df['signal'] = np.where(slow > fast, 1.0, 0.0)
    # position is the difference of the signals. Buy is 1 and Sell is -1.
    df['position'] = df['signal'].diff()
    return df


# In[10]:


def gauge(avg_pos):
    """
    Transforms the average position (int) into the respective action (string)
    of strong buy, buy, neutral, sell or strong sell
    """
    if 0.6 <= avg_pos <= 1.0:
        return 'Strong Buy'
    elif 0.2 <= avg_pos <0.6:
        return 'Buy'
    elif -0.2 <=avg_pos< 0.2:
        return 'Neutral'
    elif -0.6 <= avg_pos < -0.2:
        return 'Sell'
    elif -1.0 <= avg_pos < -0.6:
        return 'Strong Sell'


# In[11]:


def ema_strategy(data):
    """
    Computes the action given by 10 different exponential moving average-based trading strategies
    Returns:
        a list of the actions each ema strategy suggests
    """
    ema_5_200 = data['timestamp'].to_frame()
    ma_crossover(ema_5_200, data['close'], data['ema_5'], data['ema_200'])


    ema_13_100 = data['timestamp'].to_frame()
    ma_crossover(ema_13_100, data['close'], data['ema_13'], data['ema_100'])

    ema_13_200 = data['timestamp'].to_frame()
    ma_crossover(ema_13_200, data['close'], data['ema_13'], data['ema_200'])


    ema_21_100 = data['timestamp'].to_frame()
    ma_crossover(ema_21_100, data['close'], data['ema_21'], data['ema_100'])

    ema_21_200 = data['timestamp'].to_frame()
    ma_crossover(ema_21_200, data['close'], data['ema_21'], data['ema_200'])

    ema_34_55 = data['timestamp'].to_frame()
    ma_crossover(ema_34_55, data['close'], data['ema_34'], data['ema_55'])

    ema_34_100 = data['timestamp'].to_frame()
    ma_crossover(ema_34_100, data['close'], data['ema_34'], data['ema_100'])

    ema_34_200 = data['timestamp'].to_frame()
    ma_crossover(ema_34_200, data['close'], data['ema_34'], data['ema_200'])

    ema_55_100 = data['timestamp'].to_frame()
    ma_crossover(ema_55_100, data['close'], data['ema_55'], data['ema_100'])

    ema_55_200 = data['timestamp'].to_frame()
    ma_crossover(ema_55_200, data['close'], data['ema_55'], data['ema_200'])

    ema_positions = [(ema_5_200.iloc[-1,-1]),
                     (ema_13_100.iloc[-1,-1]), (ema_13_200.iloc[-1,-1]),
                     (ema_21_100.iloc[-1,-1]), (ema_21_200.iloc[-1,-1]),
                     (ema_34_55.iloc[-1,-1]), (ema_34_100.iloc[-1,-1]), (ema_34_200.iloc[-1,-1]),
                     (ema_55_100.iloc[-1,-1]), (ema_55_200.iloc[-1,-1])]

    return ema_positions


# ## Generate Trading Signals Using Ichimoku Cloud

# In[12]:


def ichimoku_strategy(data):
    """
    Computes the action given by two Ichimoku Cloud-based trading strategies
    Returns:
        a list of the actions each of the Ichimoku strategies suggests
    """
    df = data['timestamp'].to_frame()
    df['close'] = data['close']
    df['conver_base_signal'] = 0.0
    df['price_base_signal'] = 0.0
    for i in range(len(df)):
        # Conversion-Base Signal
        # If the price is above the green cloud (leading span a > leading span b) and conversion span > base span,
        # signal = 1, bullish
        if ((data['close'][i] > data['ichi_lead_a'][i])
            and (data['ichi_lead_a'][i] > data['ichi_lead_b'][i])
            and (data['ichi_conver'][i] > data['ichi_base'][i])):
                df.iloc[i, -2] = 1.0
        # If the price is below the red cloud (leading span a < leading span b) and conversion span < base span,
        # signal = -1, bearish
        elif ((data['close'][i] < data['ichi_lead_a'][i])
            and (data['ichi_lead_a'][i] < data['ichi_lead_b'][i])
            and (data['ichi_conver'][i] < data['ichi_base'][i])):
                df.iloc[i, -2] = -1.0

        # Price-Base Signal
        # If the price is above the green cloud (leading span a > leading span b) and price > base span,
        # signal = 1, bullish
        elif ((data['close'][i] > data['ichi_lead_a'][i])
            and (data['ichi_lead_a'][i] > data['ichi_lead_b'][i])
            and (data['close'][i] > data['ichi_base'][i])):
                df.iloc[i, -1] = 1.0
        # If the price is below the red cloud (leading span a < leading span b) and price < base span,
        # signal = -1, bearish
        elif ((data['close'][i] < data['ichi_lead_a'][i])
            and (data['ichi_lead_a'][i] < data['ichi_lead_b'][i])
            and (data['close'][i] < data['ichi_base'][i])):
                df.iloc[i, -1] = -1.0
    return df.iloc[-1,-2:]


# ## Gauge Chart

# In[13]:


chart_colors = {
    "values": [50, 10, 10, 10, 10, 10],
    "labels": [" ", "STRONG SELL", "SELL", "NEUTRAL", "BUY", "STRONG BUY"],
    "marker": {
        'colors': [
            'rgb(255, 255, 255)',
            'rgb(255,0,0)',
            'rgb(255,123,138)',
            'rgb(209,211,220)',
            'rgb(84,189,254)',
            'rgb(0,140,251)'
        ],
        "line": {
            "width": 0
        }
    },
    "pull":0.05,
    "domain": {'x': [0,1], 'y': [0,1]},
    "name": "Gauge",
    "hole": .85,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 90,
    "showlegend": False,
    "textinfo": "none",
    "textposition": "inside",
    "hoverinfo": "none",
}

chart_labels = {
    "values": [50, 10, 10, 10, 10, 10],
    "labels": [" ", "STRONG SELL", "SELL", "NEUTRAL", "BUY", "STRONG BUY"],
    "marker": {
        'colors': [
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)',
            'rgb(255, 255, 255)'
        ],
        "line": {
            'color': 'rgb(255, 255, 255)',
            "width": 4
        }
    },
    "domain": {'x': [0,1], 'y': [0,1]},
    "name": "Gauge",
    "hole": .90,
    "type": "pie",
    "direction": "clockwise",
    "rotation": 90,
    "showlegend": False,
    "textinfo": "label",
    "textposition": "inside",
    "hoverinfo": "none",
}

example_layout = {
    'xaxis': {
        'showticklabels': False,
        'showgrid': False,
        'zeroline': False,
    },
    'yaxis': {
        'showticklabels': False,
        'showgrid': False,
        'zeroline': False,
    },
    'font': {
        'family':'Anaheim',
        'size':20
    },
    'width':925,
    'height':925,
    'margin':{'l':0,
              'r':0,
              't':0,
              'b':0,
             },
}

dict_of_fig = dict({
    'data':[chart_colors, chart_labels],
    'layout':example_layout
})

example_fig = go.Figure(dict_of_fig)


# In[ ]:


example_fig.show()


# # Machine Learning Model

# ## Imports / Docs

# In[20]:


import models.oracle as oracle


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.accuracy')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.accuracy_since_deployment')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.smape')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.smape_since_deployment')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.noise')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.noise_since_deployment')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.tweets')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.features')


# In[15]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.predict')


# In[16]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.predict_pct')


# In[17]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.predictions_so_far')


# In[18]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.history')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.has_model')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.run')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'oracle.Oracle.stop')


# ## Example Usage

# In[ ]:


import plotly.graph_objects as go

# Remember that data such as the number of likes or retweets
# are NOT updated as time goes on. With that in mind, any
# subset of the following features are recommended when using
# the model.
twitter_features = [ 'num_tweets', 'subjectivity_sum', 'polarity_sum', 'subjectivity_avg', 'polarity_avg', 'tone_most_common' ]

predictor = oracle.Oracle(days_for_training=1, twitter_features=twitter_features)
fig = go.FigureWidget(layout=go.Layout(
        title=go.layout.Title(text="Live BTC Trading Prices")
))
fig.add_scatter(name='actual')
fig.add_scatter(name='prediction')
predictor.run(verbose=True, fig=fig)
fig


# In[ ]:


# What are the latest bitcoin tweets?
predictor.tweets()


# In[ ]:


# This returns the feature matrix being used at the current minute
predictor.features()


# In[ ]:


# This returns True if a model has been found
predictor.has_model()


# In[ ]:


# This returns the model's price forecast for the next minute
# If no model exists, this returns None
predictor.predict()


# In[ ]:


# This returns the model's price percent change forecast for the next minute
# If no model exists, this returns None
print("Predicted percent change:", predictor.predict_pct(), "%")


# In[ ]:


# This returns a dataframe of the model's predictions over the past `days_for_training` days
# If no model exists, this returns None
data = predictor.history()
data


# In[ ]:


# This will plot predictions and actual prices on the same graph
import plotly.express as px
plot = px.line(data, x="time", y=['actual', 'prediction'])
plot.show()


# In[ ]:


# This will compute metrics over the past `days_for_training` days
print("Accuracy:\t\t\t", predictor.accuracy(step=15))
print("SMAPE:\t\t\t\t" , predictor.smape(), "%")
print("Noise:\t\t\t\t" , predictor.noise())


# In[ ]:


# This will compute metrics for the past 2 hours (120 minutes) over the past `days_for_training` days
print("Accuracy:\t\t\t", predictor.accuracy(step=15, offset=120))
print("SMAPE:\t\t\t\t" , predictor.smape(offset=120), "%")
print("Noise:\t\t\t\t" , predictor.noise(offset=120))


# In[ ]:


# Run this once a few predictions have been made!
predictor.predictions_so_far()


# In[ ]:


# This will compute metrics using all predictions made so far
# NOTE: metrics will be unreliable if there haven't been many predictions
print("Accuracy:\t\t\t", predictor.accuracy_since_deployment(step=5))
print("SMAPE:\t\t\t\t" , predictor.smape_since_deployment(), "%")
print("Noise:\t\t\t\t" , predictor.noise_since_deployment())


# In[ ]:


# This will compute metrics for the past 30 minutes using the predictions made so far
# NOTE: metrics will be unreliable if there haven't been many predictions
print("Accuracy:\t\t\t", predictor.accuracy_since_deployment(step=5, offset=30))
print("SMAPE:\t\t\t\t" , predictor.smape_since_deployment(offset=30), "%")
print("Noise:\t\t\t\t" , predictor.noise_since_deployment(offset=30))


# In[ ]:


predictor.stop(wait=False)



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


#Row 2: Metrics, Trading Signals, and Twitter Sentiment Analysis
# Metrics
column_1 = [
    #Past 5 Hours
    html.H2("Metrics: Past 5 Hours"),
    html.H3(id='accuracy-recent',style={'backgroundColor':'white'}),
    html.H3(id='smape-recent',style={'backgroundColor':'white'}),
    html.H3(id='noise-recent',style={'backgroundColor':'white'}),
    #Past Day
    html.H2('Metrics: Past Day', style={'size': 50}),
    html.H3(id='accuracy-training',style={'backgroundColor':'white'}),
    html.H3(id='smape-training',style={'backgroundColor':'white'}),
    html.H3(id='noise-training',style={'backgroundColor':'white'}),
    #Predicted Percent Change
    dcc.Graph(id='percent-change'),
    dcc.Interval(id = 'metrics-update',
                 interval = 30 * 1000, #Update graph every 30 seconds (in milliseconds)
                 n_intervals = 0,
                 max_intervals = -1
                ),
]

# Trading Signals
column_2 = [
    html.H2("Trading Signals"),
    dcc.Graph(id='trading-gauge'),
    dcc.Interval(id = 'ts-update',
                 interval = 30 * 1000, #Update graph every 30 seconds (in milliseconds)
                 n_intervals = 0,
                 max_intervals = -1
                ),
]

# Twitter Sentiment Analysis
column_3 = [
    html.H2("Twitter Sentiment Analysis"),
    dcc.Graph(id='twitter-polarity'),
    dcc.Graph(id='twitter-subjectivity'),
    dcc.Interval(id = 'twitter-update',
                 interval = 30 * 1000, #Update graph every 30 seconds (in milliseconds)
                 n_intervals = 0,
                 max_intervals = -1
                )
]

body = dbc.Container(
    children = [
        html.H1('BTC Predictor',
               style={'size': 1000}
               ),
        html.Hr(),
        dbc.Row(
            children = [dbc.Col(row_1)],
            style = {'padding': '15px'}
        ),
         dbc.Row(
             children = [dbc.Col(column_1,
                                 md=3,
                                ),
                         dbc.Col(column_2,
                                 md=3,
                                ),
                         dbc.Col(column_3,
                                )
                        ],
             style = {'padding': '15px'}
        ),
    ],
    style = {'backgroundColor': colors['background'],
             'font':'Anaheim',
             'color':colors['text']
            },
    fluid = True,
)

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
#Real-Time Results: Past 5 Hours Metrics
# Accuracy
@app.callback(
    Output('accuracy-recent', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def accuracy_recent_updates(n_intervals):
    accuracy_recent = predictor.accuracy(offset=300, step=15)
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('Accuracy: {0:.2f}'.format(accuracy_recent), style=style)


# SMAPE
@app.callback(
    Output('smape-recent', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def smape_recent_updates(n_intervals):
    smape_recent = predictor.smape(offset=300)
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('SMAPE: {0:.2f}'.format(smape_recent), style=style)


# Noise
@app.callback(
    Output('noise-recent', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def noise_recent_updates(n_intervals):
    noise_recent = predictor.noise(offset=300)
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('Noise: {0:.2f}'.format(noise_recent), style=style)

#Real-Time Results: Past Day Metrics
# Accuracy
@app.callback(
    Output('accuracy-training', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def accuracy_training_updates(n_intervals):
    accuracy_training = predictor.accuracy(step=15)
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('Accuracy: {0:.2f}'.format(accuracy_training), style=style)


# SMAPE
@app.callback(
    Output('smape-training', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def smape_training_updates(n_intervals):
    smape_training = predictor.smape()
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('SMAPE: {0:.2f}'.format(smape_training), style=style)


# Noise
@app.callback(
    Output('noise-training', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def noise_training_updates(n_intervals):
    noise_training = predictor.noise()
    style = {'padding': '5px',
             'fontSize': '30px',
             'color': colors['background'],
             }
    return html.Span('Noise: {0:.2f}'.format(noise_training), style=style)

# Percent Change
@app.callback(
    Output('percent-change', 'figure'),
    [Input('metrics-update', 'n_intervals')]
)
def percent_change_updates(n_intervals):
    pct_fig = go.Figure(go.Indicator(
        mode = "number+delta",
        value = predictor.predict(),
        number = {'prefix': "$",
                  'valueformat':".00f"
                 },
        delta = {'position': "top",
                 'reference': predictor.history(1)['actual'].values[0],
                 'relative':False,
                 'valueformat':"$,.2f"
                },
        title = {"text": "Forecast"},
        ))
    pct_fig.update_layout(paper_bgcolor = "white")
    return pct_fig


#Real-Time Results: Technical Analysis
@app.callback(
    Output('trading-gauge', 'figure'),
    [Input('ts-update', 'n_intervals')]
)
def trading_signals_updates(n_intervals):
    # Every time this function is called, it returns the current time - `minutes` as a string
    get_start_time = lambda minutes: (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")

    # Ranges are inclusive, so if you want exactly 200 data points use 199 minutes
    data_real_min = d.btcstock.get_by_range(get_start_time(1439), delay=0, verbose=False)

    # Compute the indicators + add them to 'data_real_min' inplace
    fill_trends(data_real_min)

    # Combines all 15 trading strategies
    actions = np.append(ema_strategy(data_real_min), ichimoku_strategy(data_real_min)).tolist()

    # Counts the number of strategies that signals sell, buy, and neutral
    sell_num = countX(actions, x=-1)
    buy_num = countX(actions, x=1)
    neutral_num = countX(actions, x=0)

    # Computes the average trading signal and current gauge of the BTC/USD market
    overall_strength = np.mean(actions)
    overall_gauge = gauge(overall_strength)

    print(datetime.now().strftime("%H:%M:%S"))
    print('Sell: {} strategies'.format(sell_num))
    print('Buy: {} strategies'.format(buy_num))
    print('Neutral: {} strategies'.format(neutral_num))
    print('Current Strength: {}'.format(overall_strength))
    print('Current Gauge: {}'.format(overall_gauge))

    new_fig = go.Figure(example_fig)

    #Determine arrow's direction based on the current gauge of the BTC/USD market
    if (overall_gauge == 'Strong Buy'):
        theta = 41
        result_text = 'STRONG BUY'
        result_color = 'rgb(0,140,251)'
    elif (overall_gauge == 'Buy'):
        theta = 61
        result_text = 'BUY'
        result_color = 'rgb(84,189,254)'
    elif (overall_gauge == 'Neutral'):
        theta = 90
        result_text = 'NEUTRAL'
        result_color = 'rgb(209,211,220)'
    elif (overall_gauge == 'Sell'):
        theta = 118
        result_text = 'SELL'
        result_color = 'rgb(255,123,138)'
    elif (overall_gauge == 'Strong Sell'):
        theta = 136
        result_text = 'STRONG SELL'
        result_color = 'rgb(255,0,0)'

    r = 0.9
    x_head = r * cos(radians(theta))
    y_head = r * sin(radians(theta))

    #Add arrow to the gauge chart
    new_fig.add_annotation(
        ax=0,
        ay=0.5,
        axref='x',
        ayref='y',
        x=x_head,
        y=y_head,
        xref='x',
        yref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=4,
        text=result_text,
        font= {
            'family':'Anaheim',
            'color':result_color,
            'size':50
        }
    )


    new_fig.update_layout(
        font={'color': colors['text'], 'family': "Anaheim", 'size':20},
        xaxis={'showgrid': False, 'showticklabels':False, 'range':[-1,1]},
        yaxis={'showgrid': False, 'showticklabels':False, 'range':[0,1]},
        plot_bgcolor='rgba(0,0,0,0)'
        )
    return new_fig

#Real-Time Results: Twitter Sentiment Analysis Graph
@app.callback(
    Output('twitter-polarity', 'figure'),
    [Input('twitter-update', 'n_intervals')]
)
def twitter_polarity_updates(n_intervals):
    if (predictor.has_model()):
        #Features
        twitter_df = predictor.features().tail(60).reset_index(drop=True)
        print("Current Twitter Polarity: " + str(twitter_df.loc[58,'polarity_avg']))
        print("Previous Twitter Polarity: " + str(twitter_df.loc[57,'polarity_avg']))

        #Plot
        pol_fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = twitter_df.loc[58,'polarity_avg'],
            delta = {"reference": twitter_df.loc[57,'polarity_avg']
                    },
            title = {"text": "Average Polarity"},
        ))

        pol_fig.add_trace(go.Scatter(
            x=twitter_df['date'],
            y=twitter_df['polarity_avg'],
            ))

        pol_fig.update_layout(font = {'family':'Anaheim',
                                      'size':30,
                                 }
                         )
        return pol_fig



@app.callback(
    Output('twitter-subjectivity', 'figure'),
    [Input('twitter-update', 'n_intervals')]
)
def twitter_subjectivity_updates(n_intervals):
    if (predictor.has_model()):
        #Features
        twitter_df = predictor.features().tail(60).reset_index(drop=True)
        print("Current Twitter Subjectivity: " + str(twitter_df.loc[58,'subjectivity_avg']))
        print("Previous Twitter Subjectivity: " + str(twitter_df.loc[57,'subjectivity_avg']))

        #Plot
        sub_fig = go.Figure(go.Indicator(
            mode = "number+delta",
            value = twitter_df.loc[58,'subjectivity_avg'],
            delta = {"reference": twitter_df.loc[57,'subjectivity_avg']
                    },
            title = {"text": "Average Subjectivity"},
        ))

        sub_fig.add_trace(go.Scatter(
            x=twitter_df['date'],
            y=twitter_df['subjectivity_avg'],
            ))

        sub_fig.update_layout(font = {'family':'Anaheim',
                                      'size':30,
                                 }
                         )
        return sub_fig

# Run app and display result external in the notebook

app.run_server(debug=True)
