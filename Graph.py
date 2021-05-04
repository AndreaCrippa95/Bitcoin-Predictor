#normal Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web

#From previous worksheets
from Brownian_Motion import results, model
from Database import df, end, prediction_days

act = df['BTC Price']

#Create Real Values
s = end
e = end + dt.timedelta(days=prediction_days-1)
date = pd.date_range(start=s.strftime("%m/%d/%Y"), end=e.strftime("%m/%d/%Y"))
df2 = pd.DataFrame(index=date)

#Load the data for Bitcoin Price
Ticker = 'btcusd'

Price = web.get_data_tiingo(Ticker,s.strftime("%Y-%m-%d"),e.strftime("%Y-%m-%d"), api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
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
act.plot(ax=ax, color='b', label='Past')
pred.plot(ax=ax, color='r', label='Predicted')
real.plot(ax=ax, color='g', label='Actual')
ax.legend(loc='upper left')
ax.set_title('Bitcoin Price '+str(model))
plt.show()
fig.savefig('Graphs/Bitcoin_Price'+str(model)+'.png')
plt.close(fig)

