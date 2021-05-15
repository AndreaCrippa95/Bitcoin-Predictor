#Measures accuracy
import pandas as pd
import numpy as np
import pandas_datareader as web
import Inputs
import datetime as dt
from sklearn.metrics import *

start = Inputs.start
end = Inputs.end
prediction_days = Inputs.prediction_days
ACC = Inputs.ACC

#Create Real Values
if ACC:
    s = end
    e = end + dt.timedelta(prediction_days-1)

    #Load the data for Bitcoin Price
    Ticker = 'btcusd'

    Price = web.get_data_tiingo(Ticker,end.strftime("%Y-%m-%d"),e.strftime("%Y-%m-%d"), api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
    #Drops multilevel index from the Tiingo dataframe
    Price = Price.droplevel('symbol')
    #Drops TimeZone sensitivity from the Tiingo dataframe
    Price = Price.tz_localize(None)
    #Merge the closing Price with the already present dataframe keeping in ciunt the date

    y = np.array(Price['close'])
    ext = e
    while len(y)<prediction_days:
        ext = ext + dt.timedelta(+1)
        Price = web.get_data_tiingo(Ticker, end.strftime("%Y-%m-%d"), ext.strftime("%Y-%m-%d"),
                                    api_key=('eef2cf8be7666328395f2702b5712e533ea072b9'))
        Price = Price.droplevel('symbol')
        Price = Price.tz_localize(None)
        y = np.array(Price['close'])

    results = pd.read_csv('data/results',header=None)
    y_pred = np.array(results)

    R2D2 = r2_score(y,y_pred)
    Minnie = mean_squared_error(y,y_pred)
    Vodka = mean_absolute_error(y,y_pred)

    print('\nAccuracy:', file=open('data/Accuracy.txt', 'w'))
    print('-' * 80, file=open('data/Accuracy.txt', 'a'))
    print('R-squared: %s' % R2D2, file=open('data/Accuracy.txt', 'a'))
    print('Mean squared error: %s' % Minnie, file=open('data/Accuracy.txt', 'a'))
    print('Mean absolute error: %s' % Vodka, file=open('data/Accuracy.txt', 'a'))
    print('-' * 80, file=open('data/Accuracy.txt', 'a'))
