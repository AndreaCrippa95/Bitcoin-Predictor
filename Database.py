#Creating a database for the project
#Imports:
import pandas as pd
import pandas_datareader as web
import datetime as dt

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

end = dt.datetime(2020,1,1)
end = end.strftime("%Y-%m-%d")

"""
#why not written:
end = base #to have the time actualised daily ?
end = end.strftime("%Y-%m-%d")
"""

Price = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
Price = Price['close'].values

df = df.drop(df[df.index<start].index)
df = df.drop(df[df.index>end].index)
df = df.drop(df.index[[0,-1,-2]])
#February with 29 days is fucking up everything...
df['Price'] = Price
df.tail()


#GOLD

#Ticker = 'gold'

#Gold = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
#Gold = Gold['close'].values

#df['Gold'] = Gold

#S&P

#Ticker = 'ndaq'

#NDAQ = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
#NDAQ = NDAQ['close'].values

#df['NDAQ'] = NDAQ
