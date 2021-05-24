import pandas as pd
import numpy as np

import os
import sys
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data'
sys.path.append(path)

file_path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/DataFrame'

df = pd.read_csv(file_path, header=0)

df.columns.values[0] = 'Date'
df.columns.values[1] = 'BTC_Price'
df.columns.values[2] = 'Gold_Price'
df.columns.values[3] = 'NDAQ_Price'
df = df.set_index('Date')

df_X = df.loc[:,'BTC_Price']

prediction_days = 100
predictor = df_X.shift(-prediction_days)

X = np.array(df_X)
X = X[:len(df)-prediction_days]

y = np.array(predictor)
y = y[:-prediction_days]
y = y.reshape(-1,1)


def train_and_display(name):
    model = models[name]()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 10000)
    y_range = model.predict(x_range.reshape(-1, 1))

    go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
    go.Scatter(x=x_range, y=y_range, name='prediction')

    return fig
