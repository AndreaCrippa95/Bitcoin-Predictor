import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import linear_model, tree, neighbors
from sklearn.linear_model import Lasso, ElasticNet

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(path, header=0)
df.index = df.index.astype('<M8[ns]')
df = df.dropna()
prediction_days = 10

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

df = df.dropna()

predictor = df['BTC_Price'].shift(-prediction_days)

X_train = np.array(df) #x_train
X_test = X[:len(df)-prediction_days] #x_test

y_train = np.array(predictor) # y_train
y_test = y[:-prediction_days] # y_test
y = y.reshape(-1,1)


models = {'Regression': linear_model.LinearRegression,
          'Lasso': linear_model.Lasso,
          'ElasticNet': linear_model.ElasticNet,
          'Decision Tree': tree.DecisionTreeRegressor,
          'RandomForestRegressor': RandomForestRegressor,
          'GradientbosstingRegressor': GradientBoostingRegressor,
          'k-NN': neighbors.KNeighborsRegressor}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train,
                   name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test,
                   name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range,
                   name='prediction')
    ])

    return fig

app.run_server(debug=True)
