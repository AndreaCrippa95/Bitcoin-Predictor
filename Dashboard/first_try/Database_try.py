#Creating a database for the project
#Imports:
import pandas as pd
import pandas_datareader as web
import quandl as quandl
from OLD import Inputs

#test for fetching data to gitehub

#For easier setup of dates:
start = Inputs.start
end = Inputs.end
prediction_days = Inputs.prediction_days
BTC_Price = Inputs.BTC_Price
Gold_Price = Inputs.Gold_Price
NDAQ_Price = Inputs.NDAQ_Price
'''
import os
import sys
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor'
sys.path.append(os.path.abspath(path))
from Inputs import Init
start = dt.datetime(2012,1,1)
end  = dt.datetime(2019,1,1)
'''

a = start.strftime("%d/%m/%Y")
b = end.strftime("%d/%m/%Y")
date = pd.date_range(start=a, end=b)
df = pd.DataFrame(index=date)

if BTC_Price:
    #Load the data for Bitcoin Price
    Ticker = 'btcusd'
    c = start.strftime("%Y-%m-%d")
    d = end.strftime("%Y-%m-%d")

    Price = web.get_data_tiingo(Ticker,c,d, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
    #Drops multilevel index from the Tiingo dataframe
    Price = Price.droplevel('symbol')
    #Drops TimeZone sensitivity from the Tiingo dataframe
    Price = Price.tz_localize(None)
    #Merge the closing Price with the already present dataframe keeping in ciunt the date

    df = pd.merge(df,Price['close'], how='outer', left_index=True, right_index=True)
    df.rename(columns ={'close':'BTC Price'}, inplace = True)


#GOLD
if Gold_Price:
    #Gold prices from Quandl, I suppose AM is open and PM is close?
    Gold = quandl.get("LBMA/GOLD", authtoken="Ti1UcxgbNyuqmB78s14S",start_date=c, end_date=d)

    df = pd.merge(df,Gold['USD (PM)'], how='outer', left_index=True, right_index=True)

#S&P
if NDAQ_Price:
    Ticker = 'ndaq'

    NDAQ = web.get_data_tiingo(Ticker,c,d, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
    NDAQ = NDAQ.droplevel('symbol')
    NDAQ = NDAQ.tz_localize(None)
    df = pd.merge(df,NDAQ['close'], how='outer', left_index=True, right_index=True)
    df.rename(columns ={'close':'NDAQ Price'}, inplace = True)
#Remove Na values
df = df.dropna()
'''
path = 'Users/flavio/Documents/GitHub/Bitcoin-Predictor/data'
save_df = df.to_csv(path + '/DataFrame.csv')
'''
#save to csv
df.to_csv('data/DataFrame')
