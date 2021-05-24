import pandas as pd
import numpy as np

path = '/Bitcoin-Predictor/data/DataFrame'
df = pd.read_csv(path, header=0)
df = df.dropna()

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'

prediction_days = 100


predictor = df['BTC_Price'].shift(-prediction_days)
X = np.array(df)
X_mpred = X[:len(df)-prediction_days]

y = np.array(predictor)
y_mpred = y[:-prediction_days]
y_mpred = y_mpred.reshape(-1,1)


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
