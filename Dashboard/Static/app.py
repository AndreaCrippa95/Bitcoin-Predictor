import dash
import os
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from pandas_datareader import data as web
from datetime import datetime as dt

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

#import coinmarketcap_api as capi

#Descriptions
path1 = os.path.join('/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/Descriptions/BM.txt')

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

app.layout = html.Div(style={'backgroundColor': 'white', 'color': 'black'},
                    children=[

    html.H1(children='Welcome to our Dashboard',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    html.H5(children='To refresh the graphs and predictions, please do refer to our Readme.md file',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

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
    dcc.Graph(id='my-graph', style={'width': '500'}),


    dcc.Markdown(tx.First_desc_markdown,style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),

    html.Br(),
    html.H2( children='Actual Data and Graphs.',style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    html.H5(children='This is the graphic representation of our DataFrame, '
                     'which comprises the data collected since 2013 till today, '
                     'our dataframe has three main columns, the BTC_Price'
                     'the Gold_Price and finally the NASDAQ Price.'
                    ,style={'textAlign': 'center','color': 'grey','font-family': 'Helvetica'}),
    dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}),
    dcc.Graph(
            style={'width': '500'},
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

])

# For the Graphs

@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(selected_dropdown_value,'yahoo',dt(2015, 1, 1),dt.now())
    return {
        'data': [{
            'x': df.index,
            'y': df.Close
        }],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
    }

@app.callback(
    [Output(component_id='overview_graph', component_property='fig')],
    [Input(component_id='yaxis-type', component_property='value')])

def update_graph(yaxis_type):
    fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')
    return fig


# For the Descriptions
@app.callback(
    [Output(component_id='text-display',component_property='children')],
    [Input(component_id='text-input',component_property='value')])


def update_text_output_2(input_value):
    with open(path1, 'r') as Texlist1:
        content = Texlist1.read()
    return content

server = app.server
if __name__ == '__main__':
    app.run_server(
        port = 8060,
        host ='0.0.0.0')
