import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Sequential import results
from Database import df
import datetime as dt
import pandas_datareader as web

act = df['BTC Price']

#Create Real Values

date = pd.date_range(start='1/1/2021', end='3/1/2021')
df2 = pd.DataFrame(index=date)

#Load the data for Bitcoin Price
Ticker = 'btcusd'

start = dt.datetime(2021,1,1)
start = start.strftime("%Y-%m-%d")

end = dt.datetime(2021,3,1)
end = end.strftime("%Y-%m-%d")

Price = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
#Drops multilevel index from the Tiingo dataframe
Price = Price.droplevel('symbol')
#Drops TimeZone sensitivity from the Tiingo dataframe
Price = Price.tz_localize(None)
#Merge the closing Price with the already present dataframe keeping in ciunt the date

real = pd.merge(df2,Price['close'], how='outer', left_index=True, right_index=True)
real.rename(columns ={'close':'BTC Price'}, inplace = True)

prepred = pd.DataFrame(results, index=date)
pred = pd.merge(df2,prepred, how='outer', left_index=True, right_index=True)

fig, ax = plt.subplots()
act.plot(ax=ax,color='b', label = 'BTC Price (Past)')
pred.plot(ax=ax, color='r', label = 'BTC Price (Predicted)')
real.plot(ax=ax,color='g', label = 'BTC Price (Actual)')
ax.legend(loc='upper right')
ax.set_title('Bitcoin Price prediction')
plt.show()
