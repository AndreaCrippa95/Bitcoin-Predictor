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
        self.end = end
        self.result = result
        self.model = ChModel
        e = self.end + dt.timedelta(self.days - 1)
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
        self.df.rename(columns={'BTC Price': 'Past Price'}, inplace=True)
        act = pd.DataFrame(self.df['Past Price'])
        # Merge the closing Price with the already present dataframe keeping in ciunt the date
        real = pd.merge(self.df2, self.Price['close'], how='outer', left_index=True, right_index=True)
        real.rename(columns={'close': 'Real Price'}, inplace=True)
        prepred = pd.DataFrame(self.result, index=self.date)
        pred = pd.merge(self.df2, prepred, how='outer', left_index=True, right_index=True)
        pred.rename(columns={0: 'Prediction'}, inplace=True)

        Final = pd.merge(act, pred, how='outer', left_index=True, right_index=True)
        Final = pd.merge(Final, real, how='outer', left_index=True, right_index=True)
        Final = Final.tail(self.days + 5)

        fig, ax = plt.subplots()
        Final.plot(ax=ax)
        ax.legend(loc='upper left')
        ax.set_title('Bitcoin Price ' + str(self.model))
        plt.show()
        fig.savefig('Graphs/Bitcoin_Price' + str(self.model) + '.png')
        plt.close(fig)

    def Results(self):
        prepred = pd.DataFrame(self.result, index=self.date)
        pred = pd.merge(self.df2, prepred, how='outer', left_index=True, right_index=True)
        pred.rename(columns={0: 'Prediction'}, inplace=True)

        print('\nResults:', file=open('data/Result.txt', 'w'))
        print('{:<10}{:>13}'.format('Date', 'BTC Price'), file=open('data/Result.txt', 'a'))
        print('-' * 80, file=open('data/Result.txt', 'a'))
        print(pred['Prediction'], file=open('data/Result.txt', 'a'))
        print('-' * 80, file=open('data/Result.txt', 'a'))

    def Accuracy(self):
        # Merge the closing Price with the already present dataframe keeping in ciunt the date
        y = np.array(self.Price['close'])
        ext = self.end + dt.timedelta(self.days - 1)
        while len(y) < self.days:
            ext = ext + dt.timedelta(+1)
            Price = web.get_data_tiingo(self.Ticker, self.end.strftime("%Y-%m-%d"), ext.strftime("%Y-%m-%d"),
                                        api_key=('eef2cf8be7666328395f2702b5712e533ea072b9'))
            Price = Price.droplevel('symbol')
            Price = Price.tz_localize(None)
            y = np.array(Price['close'])

        y_pred = np.array(self.result)
        R2D2 = r2_score(y, y_pred)
        Minnie = mean_squared_error(y, y_pred)
        Vodka = mean_absolute_error(y, y_pred)

        print('\nAccuracy:', file=open('data/Accuracy.txt', 'w'))
        print('-' * 80, file=open('data/Accuracy.txt', 'a'))
        print('R-squared: %s' % R2D2, file=open('data/Accuracy.txt', 'a'))
        print('Mean squared error: %s' % Minnie, file=open('data/Accuracy.txt', 'a'))
        print('Mean absolute error: %s' % Vodka, file=open('data/Accuracy.txt', 'a'))
        print('-' * 80, file=open('data/Accuracy.txt', 'a'))
