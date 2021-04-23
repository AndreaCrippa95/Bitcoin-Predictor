#https://www.kaggle.com/abasov/bitcoin-price-prediction-with-keras/comments

import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from matplotlib import pyplot as plt
import datetime, time


#model keras found online
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout,AveragePooling1D,Reshape

from sklearn.metrics import mean_absolute_error

#Create standard DataFrame
base = dt.datetime.today()
date_list = [base - dt.timedelta(days=x) for x in range(5000)]
date_list = [date_list[x].strftime("%Y-%m-%d") for x in range(len(date_list))]
date_list = [i for i in reversed(date_list)]
df = pd.DataFrame(index=date_list)
#test for fetching data to gitehub

#Load the data for Bitcoin Price
Ticker = 'btcusd'

start = dt.datetime(2012,1,1)
start = start.strftime("%Y-%m-%d")
todaysDate1 = datetime.date.today()
end = dt.datetime(2021,4,22)
end = end.strftime("%Y-%m-%d")

Price = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
Price.tail()

len(Price['open'])
