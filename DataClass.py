#Creating a database for the project
#Imports:
import pandas as pd
import pandas_datareader as web
import quandl as quandl
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

#test for fetching data to gitehub

#For easier setup of dates:
class Data:
    def __init__(self,start,end,days,BTC,Gold,NDAQ,Returns):
        self.start = start
        self.end = end
        self.prediction_days = days
        self.BTC_Price = BTC
        self.Gold_Price = Gold
        self.NDAQ_Price = NDAQ
        self.Returns = Returns
        self.TestMode = True

        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.df = None
        self.scaler = MinMaxScaler()
        self.X_scaler = None
        self.y_scaler = None
        self.nvariables = self.BTC_Price+self.Gold_Price+self.NDAQ_Price+self.Returns

    def create_data(self):
        a = self.start.strftime("%d/%m/%Y")
        b = self.end.strftime("%d/%m/%Y")
        c = self.start.strftime("%Y-%m-%d")
        d = self.end.strftime("%Y-%m-%d")
        date = pd.date_range(start=a, end=b)
        self.df = pd.DataFrame(index=date)

        if not self.TestMode:
            # Load the data for Bitcoin Price
            Ticker = 'btcusd'
            Price = web.get_data_tiingo(Ticker, c, d, api_key=('eef2cf8be7666328395f2702b5712e533ea072b9'))
            # Drops multilevel index from the Tiingo dataframe
            Price = Price.droplevel('symbol')
            # Drops TimeZone sensitivity from the Tiingo dataframe
            Price = Price.tz_localize(None)
            # Merge the closing Price with the already present dataframe keeping in ciunt the date
            Price.to_csv('data/Price')
        else:
            Price = pd.read_csv('data/Price')
            Price['date'] = Price['date'].astype('<M8[ns]')
            Price.index = Price['date']

        self.df = pd.merge(self.df, Price['close'], how='outer', left_index=True, right_index=True)
        self.df.rename(columns={'close': 'BTC Price'}, inplace=True)

        #GOLD
        if self.Gold_Price:
            if not self.TestMode:
            #Gold prices from Quandl, I suppose AM is open and PM is close?
                Gold = quandl.get("LBMA/GOLD", authtoken="Ti1UcxgbNyuqmB78s14S",start_date=c, end_date=d)
                Gold.to_csv('data/Gold')
            else:
                Gold = pd.read_csv('data/Gold')
                Gold['Date'] = Gold['Date'].astype('<M8[ns]')
                Gold.index = Gold['Date']
            self.df = pd.merge(self.df,Gold['USD (PM)'], how='outer', left_index=True, right_index=True)

        #S&P
        if self.NDAQ_Price:
            if not self.TestMode:
                Ticker = 'ndaq'

                NDAQ = web.get_data_tiingo(Ticker,c,d, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))
                NDAQ = NDAQ.droplevel('symbol')
                NDAQ = NDAQ.tz_localize(None)
                NDAQ.to_csv('data/NDAQ')
            else:
                NDAQ = pd.read_csv('data/NDAQ')
                NDAQ['date'] = NDAQ['date'].astype('<M8[ns]')
                NDAQ.index = NDAQ['date']
            self.df = pd.merge(self.df,NDAQ['close'], how='outer', left_index=True, right_index=True)
            self.df.rename(columns ={'close':'NDAQ Price'}, inplace = True)

        if self.Returns:
            Returns = []
            Index = []
            Returns.append(0)
            Index.append(self.df.index[0])
            for i in range(len(self.df)-1):
                Returns.append((self.df['BTC Price'][i+1]-self.df['BTC Price'][i])/self.df['BTC Price'][i+1])
                Index.append(self.df.index[i+1])
            Returns = pd.DataFrame(Returns,index=Index)
            self.df = pd.merge(self.df,Returns, how='outer', left_index=True, right_index=True)
            self.df.rename(columns ={0:'Returns'}, inplace = True)

        #Remove Na values
        self.df = self.df.dropna()

        if self.TestMode:
                self.df = self.df[self.df.index <= self.end]

        if not self.BTC_Price:
            self.df = self.df.drop(columns='BTC Price')

        if self.BTC_Price:
            predictor = self.df['BTC Price'].shift(-self.prediction_days)
        elif self.Returns:
            predictor = self.df['Returns'].shift(-self.prediction_days)
        else:
            predictor = self.df['BTC Price'].shift(-self.prediction_days)

        self.X_tr = np.array(self.df)
        self.X_tr = self.X_tr[:len(self.df) - self.prediction_days]
        self.X_scaler = self.scaler
        self.X_scaler.fit(self.X_tr)
        self.X_tr = self.X_scaler.transform(self.X_tr)

        self.y_tr = np.array(predictor)
        self.y_tr = self.y_tr[:-self.prediction_days]
        self.y_tr = self.y_tr.reshape(-1, 1)
        self.y_scaler = self.scaler
        self.y_scaler.fit(self.y_tr)
        self.y_tr = self.y_scaler.transform(self.y_tr)

        self.X_te = np.array(self.df[-self.prediction_days:])
        self.X_te = self.X_scaler.transform(self.X_te.reshape(-1,1))
        self.X_te = self.X_te.reshape(self.prediction_days,self.nvariables)
