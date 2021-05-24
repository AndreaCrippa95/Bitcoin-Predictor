import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from sklearn.metrics import *

class Results:

    def __init__(self,df,result,ChModel,end,days):
        self.df = df
        self.days = days
        self.end = end + dt.timedelta(1)
        self.result = result
        self.model = ChModel
        e = self.end + dt.timedelta(self.days-1)
        b = self.end.strftime("%m/%d/%Y")
        c = e.strftime("%m/%d/%Y")
        self.date = pd.date_range(start=b, end=c)
        self.df2 = pd.DataFrame(index=self.date)
        self.Ticker = 'btcusd'
        self.Price = web.get_data_tiingo(self.Ticker, end.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"),
                                    api_key=('eef2cf8be7666328395f2702b5712e533ea072b9'))
        # Drops multilevel index from the Tiingo dataframe
        self.Price = self.Price.droplevel('symbol')
        # Drops TimeZone sensitivity from the Tiingo dataframe
        self.Price = self.Price.tz_localize(None)

    def Graph(self):
        # Merge the closing Price with the already present dataframe keeping in the date
        if self.df.columns[0] in 'BTC Price':
            real = pd.merge(self.df2, self.Price['close'], how='outer', left_index=True, right_index=True)
            real.rename(columns={'close': 'Real'}, inplace=True)
        elif self.df.columns[0] in 'Returns':
            Returns = []
            Returns.append(0)
            for i in range(len(self.date) - 1):
                Returns.append((self.Price['close'][i + 1] - self.Price['close'][i]) / self.Price['close'][i + 1])
            real = pd.DataFrame(Returns, index=self.date)
            real.rename(columns={0: 'Real'}, inplace=True)
        else:
            raise ValueError
        self.df.rename(columns={self.df.columns[0]: 'Past'}, inplace=True)
        act = pd.DataFrame(self.df['Past'])
        prepred = pd.DataFrame(self.result, index=self.date)
        pred = pd.merge(self.df2, prepred, how='outer', left_index=True, right_index=True)
        pred.rename(columns={0: 'Prediction'}, inplace=True)

        Final = pd.merge(act, pred, how='outer', left_index=True, right_index=True)
        Final = pd.merge(Final, real, how='outer', left_index=True, right_index=True)
        Final = Final.tail(self.days + 5)

        fig, ax = plt.subplots()
        Final.plot(ax=ax)
        ax.legend(loc='upper left')
        ax.set_title('Bitcoin ' + str(self.model))
        plt.show()
        fig.savefig('Graphs/Bitcoin_' + str(self.model) + '.png')
        plt.close(fig)

    def Results(self):
        prepred = pd.DataFrame(self.result, index=self.date)
        pred = pd.merge(self.df2, prepred, how='outer', left_index=True, right_index=True)
        pred.rename(columns={0: 'Prediction'}, inplace=True)

        print('\nResults:', file=open('data/Result.txt', 'w'))
        print('{:<10}{:>13}'.format('Date', 'BTC'), file=open('data/Result.txt', 'a'))
        print('-' * 80, file=open('data/Result.txt', 'a'))
        print(pred['Prediction'], file=open('data/Result.txt', 'a'))
        print('-' * 80, file=open('data/Result.txt', 'a'))
