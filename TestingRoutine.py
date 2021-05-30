import datetime as dt
import numpy as np
import pandas as pd
from DataClass import Data
from MethodClass import Method
from sklearn.metrics import *
import random
import time
import threading




#Importing once the Database to not overload Tiingo
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,5,1)
prediction_days = 10
#Choose the input data
BTC_Price = True
Gold_Price = False
NDAQ_Price = False
Returns = False
#Choose a model:
#ChModel = 'LR'
#Choose the desired output
RES = True
GRA = False

dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
dat.create_data()
df = dat.df

f = open('Testing_Routine_Results.txt', 'w')  # open file for output

for h in range(5,20,5):
    prediction_days = h
    print('Testing with the following conditions: \n'
          'Prediction Days: %s \n'
          'BTC Price: %s \n'
          'Gold Price: %s \n'
          'Nasdaq Price: %s \n'
          'BTC Returns: %s \n' % (prediction_days,BTC_Price,Gold_Price,NDAQ_Price,Returns),
          file=open('Testing_Routine_Results.txt', 'a'))

    for ChModeli in ['BM','RFR','GBR','LR','Lasso','KNR','EN','DTR','Sequential','SVM','DNN']:
        if ChModeli == 'BM':
            print('Starting with Brownian Motion')
            lim = 100
        elif ChModeli == 'Sequential':
            print('Starting with Sequential')
            lim = 5
        elif ChModeli in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
            print('Starting with Machine Learning Methods')
            lim = 100
        elif ChModeli in ['SVM']:
            print('Starting with Support Vector Machines')
            lim = 100
        elif ChModeli in ['DNN']:
            print('Starting with Dense Neural Network')
            lim = 5
        Y_TRUE = []
        Y_PRED = []
        i = 0
        while i < lim:
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
                Y_PRED.append(met.Brownian_Motion())
            elif ChModeli == 'Sequential':
                Y_PRED.append(met.Sequential())
            elif ChModeli in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
                Y_PRED.append(met.MachineLearning())
            elif ChModeli in ['SVM']:
                Y_PRED.append(met.SVM())
            elif ChModeli in ['DNN']:
                Y_PRED.append(met.DNN())

            rand = randomdate + dt.timedelta(prediction_days)
            df3 = df[df.index<rand].tail(prediction_days)
            Y_TRUE.append(np.array(df3[df3.columns[0]]))
            i += 1

        print('\nAccuracy of ' + ChModeli, file=open('Testing_Routine_Results.txt', 'a'))
        print('-' * 80, file=open('Testing_Routine_Results.txt', 'a'))
        print('R-squared: %s' % r2_score(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)), file=open('Testing_Routine_Results.txt', 'a'))
        print('Mean squared error: %s' % mean_squared_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)), file=open('Testing_Routine_Results.txt', 'a'))
        print('Mean absolute error: %s' % mean_absolute_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)), file=open('Testing_Routine_Results.txt', 'a'))
        print('-' * 80, file=open('Testing_Routine_Results.txt', 'a'))


#######################################################
#Tuning
from sklearn.linear_model import Lasso,ElasticNet,LinearRegression


model = Lasso()
model.fit(dat.X_tr,dat.y_tr)
model.predict(dat.X_te)

model = LinearRegression()
model.fit(dat.X_tr,dat.y_tr)
model.predict(dat.X_te)