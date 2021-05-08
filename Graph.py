#normal Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web

#From previous worksheets
#import os
#import sys
#path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor'
#sys.path.append(os.path.abspath(path))
from MachineLearning import model

#From Input
global end
global start
global prediction_days

df = pd.read_csv('data/DataFrame',index_col=0)
df.rename(columns ={'BTC Price':'Past Price'}, inplace = True)
df.index = df.index.astype('<M8[ns]')
act = pd.DataFrame(df['Past Price'])


#Create Real Values
s = end
e = end + dt.timedelta(prediction_days-1)
a = start.strftime("%m/%d/%Y")
b = s.strftime("%m/%d/%Y")
c = e.strftime("%m/%d/%Y")
date = pd.date_range(start=b, end=c)
df2 = pd.DataFrame(index=date)

#Load the data for Bitcoin Price
Ticker = 'btcusd'

Price = web.get_data_tiingo(Ticker,end.strftime("%Y-%m-%d"),e.strftime("%Y-%m-%d"), api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
#Drops multilevel index from the Tiingo dataframe
Price = Price.droplevel('symbol')
#Drops TimeZone sensitivity from the Tiingo dataframe
Price = Price.tz_localize(None)
#Merge the closing Price with the already present dataframe keeping in ciunt the date

real = pd.merge(df2,Price['close'], how='outer', left_index=True, right_index=True)
real.rename(columns ={'close':'Real Price'}, inplace = True)

results = pd.read_csv('data/results',header=None)
results = np.array(results)
prepred = pd.DataFrame(results, index=date)
pred = pd.merge(df2,prepred, how='outer', left_index=True, right_index=True)
pred.rename(columns ={0:'Prediction'}, inplace = True)

Final = pd.merge(act,pred, how='outer', left_index=True, right_index=True)
Final = pd.merge(Final,real, how='outer', left_index=True, right_index=True)

fig, ax = plt.subplots()
Final.plot(ax=ax)
ax.legend(loc='upper left')
ax.set_title('Bitcoin Price '+str(model))
plt.show()
fig.savefig('Graphs/Bitcoin_Price'+str(model)+'.png')
plt.close(fig)
