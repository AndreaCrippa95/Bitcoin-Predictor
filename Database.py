#Creating a database for the project
#Imports:
import pandas as pd
import pandas_datareader as web
import datetime as dt
import quandl as quandl

#test for fetching data to gitehub

date = pd.date_range(start='1/1/2020', end='1/1/2021')
df = pd.DataFrame(index=date)

#Load the data for Bitcoin Price
Ticker = 'btcusd'

start = dt.datetime(2020,1,1)
start = start.strftime("%Y-%m-%d")

end = dt.datetime(2021,1,1)
end = end.strftime("%Y-%m-%d")

"""
#why not written:
end = base #to have the time actualised daily ?
end = end.strftime("%Y-%m-%d")
"""

Price = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
#Drops multilevel index from the Tiingo dataframe
Price = Price.droplevel('symbol')
#Drops TimeZone sensitivity from the Tiingo dataframe
Price = Price.tz_localize(None)
#Merge the closing Price with the already present dataframe keeping in ciunt the date

df = pd.merge(df,Price['close'], how='outer', left_index=True, right_index=True)
df.rename(columns ={'close':'BTC Price'}, inplace = True)


#GOLD

#Gold prices from Quandl, I suppose AM is open and PM is close?
Gold = quandl.get("LBMA/GOLD", authtoken="Ti1UcxgbNyuqmB78s14S",start_date=start, end_date=end)

df = pd.merge(df,Gold['USD (PM)'], how='outer', left_index=True, right_index=True)

#S&P

Ticker = 'ndaq'

NDAQ = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
NDAQ = NDAQ.droplevel('symbol')
NDAQ = NDAQ.tz_localize(None)
df = pd.merge(df,NDAQ['close'], how='outer', left_index=True, right_index=True)
df.rename(columns ={'close':'NDAQ Price'}, inplace = True)

df = df.dropna()
