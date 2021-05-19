#Creating a database for the project
#Imports:
import pandas as pd
import pandas_datareader as web
import datetime as dt
import quandl as quandl
import Inputs
import sys
import os


#test for fetching data to gitehub

#For easier setup of dates:
class Data:
    def __init__(self,start,end,days,BTC,Gold,NDAQ):
        self.start = start
        self.end = end
        self.prediction_days = days
        self.BTC_Price = BTC
        self.Gold_Price = Gold
        self.NDAQ_Price = NDAQ

    def create_data(self):
        a = self.start.strftime("%d/%m/%Y")
        b = self.end.strftime("%d/%m/%Y")
        c = self.start.strftime("%Y-%m-%d")
        d = self.end.strftime("%Y-%m-%d")
        date = pd.date_range(start=a, end=b)
        df = pd.DataFrame(index=date)

        if self.BTC_Price:
            # Load the data for Bitcoin Price
            Ticker = 'btcusd'
            Price = web.get_data_tiingo(Ticker, c, d, api_key=('eef2cf8be7666328395f2702b5712e533ea072b9'))
            # Drops multilevel index from the Tiingo dataframe
            Price = Price.droplevel('symbol')
            # Drops TimeZone sensitivity from the Tiingo dataframe
            Price = Price.tz_localize(None)
            # Merge the closing Price with the already present dataframe keeping in ciunt the date

            df = pd.merge(df, Price['close'], how='outer', left_index=True, right_index=True)
            df.rename(columns={'close': 'BTC Price'}, inplace=True)

        #GOLD
        if self.Gold_Price:
            #Gold prices from Quandl, I suppose AM is open and PM is close?
            Gold = quandl.get("LBMA/GOLD", authtoken="Ti1UcxgbNyuqmB78s14S",start_date=c, end_date=d)

            df = pd.merge(df,Gold['USD (PM)'], how='outer', left_index=True, right_index=True)

        #S&P
        if self.NDAQ_Price:
            Ticker = 'ndaq'

            NDAQ = web.get_data_tiingo(Ticker,c,d, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
            NDAQ = NDAQ.droplevel('symbol')
            NDAQ = NDAQ.tz_localize(None)
            df = pd.merge(df,NDAQ['close'], how='outer', left_index=True, right_index=True)
            df.rename(columns ={'close':'NDAQ Price'}, inplace = True)
        #Remove Na values
        df = df.dropna()
        return df
