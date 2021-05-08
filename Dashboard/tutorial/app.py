import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/BTC_Historic/bitcoin_price_1week_Test - Test.csv'
df = pd.read_csv(path)
df.index = pd.to_datetime(df['Date'])
df.head()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#From here onwards you have to define all the functions parts of your dash
fig = px.line(x=df["Date"], y=df["Close"])

dcc.Dropdown(
    options=[
        {'label': 'Neural Network', 'value': 'RNN'},
        {'label': 'Random_walk', 'value': 'RW'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value='RNN'
)

dcc.Dropdown(
    options=[
        {'label': 'Neural Network', 'value': 'RNN'},
        {'label': 'Random_walk', 'value': 'RW'},
        {'label': 'San Francisco', 'value': 'SF'}
    ],
    value='RNN'
)
#from here onwards you have to define the modification done by the user

@app.callback


#from here onwards you have to define the layout of the Dashb
app.layout = html.Div(children=[
    html.H1(children='Welcome to our Bitcoin Forecasting Dashboard'),
    html.Div(children='The forecast in this blog are for general informational purposes only and are not intended to provide specific advice financial advise.'),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    dcc.Dropdown(id='to define')
])

#from here onwards you have to define on which server you want to define your dash
server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
