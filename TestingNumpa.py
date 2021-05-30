import datetime as dt
import numpy as np
import pandas as pd
from numba import njit,jit
from DataClass import Data
from MethodClass import Method
from sklearn.metrics import *
import random

global Data
global Method

#Importing once the Database to not overload Tiingo
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,5,1)
prediction_days = 10
#Choose the input data
BTC_Price = False
Gold_Price = False
NDAQ_Price = False
Returns = True
#Choose a model:
ChModel = 'LR'
#Choose the desired output
RES = True
GRA = False

dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
dat.create_data()
df = dat.df

@jit()
def Testing():
    Final = np.zeros(3)
    for h in range(5,20,5):
        prediction_days = h
        Accuracy = np.zeros((11,3))
        k = 0
        for ChModeli in ['BM','RFR','GBR','LR','Lasso','KNR','EN','DTR','Sequential','SVM','DNN']:
            if ChModeli == 'BM':
                lim = 100
            elif ChModeli == 'Sequential':
                lim = 5
            elif ChModeli in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
                lim = 100
            elif ChModeli in ['SVM']:
                lim = 100
            elif ChModeli in ['DNN']:
                lim = 5

            Y_TRUE = np.zeros(lim*h)
            Y_PRED = np.zeros(lim*h)

            for i in range(0,lim*h,h):
                start_date = dt.date(2012, 2, 1)
                end_date = dt.date(2020, 12, 31)

                time_between_dates = end_date - start_date
                days_between_dates = time_between_dates.days
                random_number_of_days = random.randrange(days_between_dates)
                randomdate = start_date + dt.timedelta(days=random_number_of_days)
                randomdate = pd.to_datetime(randomdate)
                randomdate = randomdate.to_pydatetime()

                dat2 = Data(start_date,randomdate,prediction_days,BTC_Price,Gold_Price,NDAQ_Price,Returns)
                dat2.create_data()
                df2 = dat2.df
                met = Method(df2,ChModel=ChModeli,days=prediction_days,Data=dat2)
                if ChModeli == 'BM':
                    Y_PRED[i:i+h] = np.array(met.Brownian_Motion()).reshape(h)
                elif ChModeli == 'Sequential':
                    Y_PRED[i:i+h] =  np.array(met.Sequential()).reshape(h)
                elif ChModeli in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
                    Y_PRED[i:i+h] = np.array(met.MachineLearning()).reshape(h)
                elif ChModeli in ['SVM']:
                    Y_PRED[i:i+h] = np.array(met.SVM()).reshape(h)
                elif ChModeli in ['DNN']:
                    Y_PRED[i:i+h] = np.array(met.DNN()).reshape(h)

                rand = randomdate + dt.timedelta(prediction_days)
                df3 = df[df.index<rand].tail(prediction_days)
                Y_TRUE[i:i+h] = np.array(df3[df3.columns[0]])


            Accuracy[k] = np.array([r2_score(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)),
                           mean_squared_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)),
                           mean_absolute_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1))])
            k += 1

        Final = np.append(Final,Accuracy)
    return Final
