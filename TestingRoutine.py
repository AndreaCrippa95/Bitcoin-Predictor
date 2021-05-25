import datetime as dt
import numpy as np
import pandas as pd
from DataClass import Data
from MethodClass import Method
from sklearn.metrics import *
import random


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
ChModel = 'LR'
#Choose the desired output
RES = True
GRA = False

dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
dat.create_data()
df = dat.df
#X_tr = dat.X_tr
#X_te = dat.X_te
#y_tr = dat.y_tr
#df = pd.DataFrame(df)
#df.index = pd.to_datetime(df.index)
#######
#met = Method(df,ChModel=ChModel,days=prediction_days,Data=dat)
#y_pred = met.Sequential()
######
#Randomize starting date:
Y_TRUE = []
Y_PRED = []
i = 0
while i < 1000:
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
    met = Method(df2,ChModel=ChModel,days=prediction_days,Data=dat2)
    Y_PRED.append(met.MachineLearning())

    rand = randomdate + dt.timedelta(prediction_days)
    df3 = df[df.index<rand].tail(prediction_days)
    Y_TRUE.append(np.array(df3[df3.columns[0]]))
    i += 1

print('\nAccuracy:')
print('-' * 80)
print('R-squared: %s' % r2_score(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)))
print('Mean squared error: %s' % mean_squared_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)))
print('Mean absolute error: %s' % mean_absolute_error(np.array(Y_TRUE).reshape(-1,1),np.array(Y_PRED).reshape(-1,1)))
print('-' * 80)

'''
def eval_on_features2(X_train, X_test, y_train, y_test, regressor):
    regressor.fit(X_train, y_train)
    print("Test-set R^2 train: {:.2f}".format(regressor.score(X_train, y_train)))
    print("Test-set R^2 test: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    plt.xticks(pd.date_range(start="01/01/2019", end="31/12/2020", freq='D').astype("int"), rotation=90, ha="left")

    plt.plot(range(len(X_train)), y_train, label="train")
    plt.plot(range(len(X_train), len(y_test) + len(X_train)), y_test, '-', label="test")
    plt.plot(range(len(X_train)), y_pred_train, '--', label="prediction train")

    plt.plot(range(len(X_train), len(y_test) + len(X_train)), y_pred, '--', label="prediction test")

    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("BTC")

for i in np.array([50,100,200,300,400,500]):
    print("number of estimators: ",i)
    regressor = GradientBoostingRegressor(n_estimators=i, random_state=0)
    eval_on_features2(X_tr,X_te,y_tr,y_te,regressor)

'''
