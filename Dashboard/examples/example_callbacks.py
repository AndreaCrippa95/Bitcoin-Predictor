from sklearn.linear_model import LinearRegression
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(path, header=0)
df.columns.values[0] = 'Date'
df.columns.values[2] = 'Gold'
df.columns.values[3] = 'NASDAQ'

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='year-slider',
        min=df['Date'].min(),
        max=df['Date'].max(),
        value=df['Date'].min(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('year-slider', 'value'))
def update_figure(selected_year):
    filtered_df = df[df.Date == selected_year]

    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
                     size="pop", color="continent", hover_name="country",
                     log_x=True, size_max=55)

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

server = app.server
if __name__ == '__main__':
    app.run_server(
        port=8060,
        host='0.0.0.0')
