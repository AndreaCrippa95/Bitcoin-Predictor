import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn import linear_model

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

file_path = '/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(file_path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'
df = df.set_index('Date')

df_X = np.array(df.loc[:,'Gold_Price'].values)
X = df_X.reshape(-1,1)
df_Y = np.array(df.loc[:,'BTC_Price'].values)
y = df_Y.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

models = {'Regression': linear_model.LinearRegression,
          'Lasso': linear_model.Lasso}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.P("Select Model:"),
    dcc.Dropdown(
        id='model-name',
        options=[{'label': x, 'value': x}
                 for x in models],
        value='Regression',
        clearable=False
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"),
    [Input('model-name', "value")])
def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = df.loc[:,'Date'].values
    y_range = np.linspace(X.min(), X.max(), 10000)
    fig = go.Figure([

        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='prediction')])

    return fig

app.run_server(debug=True)
