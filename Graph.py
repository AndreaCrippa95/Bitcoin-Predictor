import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Sequential import results
from Database import df
import datetime as dt
import pandas_datareader as web

date = pd.date_range(start='1/1/2007', end='1/1/2021')
act = df.columns[0]
#Create Real Values
Ticker = 'btcusd'
start = dt.datetime(2000,1,1)
start = start.strftime("%Y-%m-%d")
end = dt.date.today()
end = end.strftime("%Y-%m-%d")
real = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
real = real['close']


fig, ax = plt.subplots()
ax(act, date, color='b', label = 'BTC Price (Past)')
ax(results, date, color='r', label = 'BTC Price (Predicted)')
ax(real, date, color='g', label = 'BTC Price (Actual')
ax.legend(loc='upper right')
ax.set_title('Bitcoin Price prediction')
plt.show()
