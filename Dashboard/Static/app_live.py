import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from pandas_datareader import data as web
from datetime import datetime as dt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Bitcoin', 'value': 'BTC-USD'},
            {'label': 'Ripple', 'value': 'XRP-USD'},
            {'label': 'Matic', 'value': 'MATIC-USD'},
            {'label': 'Binance ', 'value': 'BNB-USD'},
            {'label': 'Moon', 'value': 'MOON.SW'},
            {'label': 'Etheureum', 'value': 'ETH-USD'},
            {'label': 'Litecoin', 'value': 'LTC-USD'}
        ],
        value='MOON.SW'
    ),
    dcc.Graph(id='my-graph')
], style={'width': '500'})

@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(
        selected_dropdown_value,
        'yahoo',
        dt(2015, 1, 1),
        dt.now()
    )
    return {'data': [{'x': df.index,'y': df.Close}],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}}

if __name__ == '__main__':
    app.run_server()
