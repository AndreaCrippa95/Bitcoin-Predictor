import datetime as dt
import numpy as np
import pandas as pd
from DataClass import Data
from MethodClass import Method
from sklearn.metrics import *
import random

#Importing once the Database to not overload Tiingo
start = dt.datetime(2012,1,1)
end = dt.date.today()
prediction_days = 10
#Choose the input data
BTC_Price = True
Gold_Price = False
NDAQ_Price = False
#Choose a model:
ChModel = 'LR'
#Choose the desired output
RES = True
GRA = True
ACC = True
dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price)
dat.create_data()
df = dat.df
df = pd.DataFrame(df)
df.index = pd.to_datetime(df.index)
#######
met = Method(df,ChModel=ChModel,days=prediction_days,Data=dat)
y_pred = met.MachineLearning()
######
#Randomize starting date:
R2D2 = []
Minnie = []
Vodka = []
i = 0
while i < 25:
    start_date = dt.date(2012, 2, 1)
    end_date = dt.date(2020, 12, 31)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    randomdate = start_date + dt.timedelta(days=random_number_of_days)
    randomdate = pd.to_datetime(randomdate)
    df2 = df[df.index<randomdate]
    met = Method(df2,ChModel=ChModel,days=prediction_days,Data=dat)
    y_pred = met.MachineLearning()

    rand = randomdate + dt.timedelta(prediction_days)
    df3 = df[df.index<rand].tail(prediction_days)
    y_true = np.array(df3['BTC Price'])

    R2D2.append(r2_score(y_true, y_pred))
    Minnie.append(mean_squared_error(y_true, y_pred))
    Vodka.append(mean_absolute_error(y_true, y_pred))
    i += 1

print('\nAccuracy:')
print('-' * 80)
print('R-squared: %s' % np.mean(R2D2))
print('Mean squared error: %s' % np.mean(Minnie))
print('Mean absolute error: %s' % np.mean(Vodka))
print('-' * 80)
