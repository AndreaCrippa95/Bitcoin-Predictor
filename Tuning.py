import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from DataClass import Data
import datetime as dt
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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
    plt.ylabel("DATL")

start = dt.datetime(2012,1,1)
end = dt.date.today()
prediction_days = 10
#Choose the input data
BTC_Price = True
Gold_Price = False
NDAQ_Price = False
Returns = False
#Choose a model:
ChModel = 'DNN'
#Choose the desired output
RES = True
GRA = True

dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
dat.create_data()
df = dat.df
df = pd.DataFrame(df)
df.index = pd.to_datetime(df.index)
DATE = dt.datetime(2015,1,1)
df2 = df[df.index < DATE]
rand = DATE + dt.timedelta(prediction_days)
df3 = df[df.index<rand].tail(prediction_days)
y_true = np.array(df3['BTC Price'])

predictor = df2['BTC Price'].shift(-prediction_days)

X_tr = np.array(df2)
X_tr = X_tr[:len(df2) - prediction_days]
scaler = MinMaxScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)

y_tr = np.array(predictor)
y_tr = y_tr[:-prediction_days]
y_tr = y_tr.reshape(-1, 1)
y_tr = scaler.transform(y_tr)

X_te = np.array(df[-prediction_days:])
X_te = scaler.transform(X_te)

pipelines = []
pipelines.append(('LR', Pipeline([('LR', LinearRegression())])))
pipelines.append(('LASSO', Pipeline([('LASSO', Lasso())])))
pipelines.append(('EN', Pipeline([('EN', ElasticNet())])))
pipelines.append(('KNN', Pipeline([('KNN', KNeighborsRegressor())])))
pipelines.append(('CART', Pipeline([('CART', DecisionTreeRegressor())])))
pipelines.append(('GBM', Pipeline([('GBM', GradientBoostingRegressor())])))
pipelines.append(('RFR', Pipeline([('RFR', RandomForestRegressor())])))

results = []
names = []
i = -1
for name, model in pipelines:
    i=i+1
    kfold = KFold(n_splits=10)
    model.fit(X_tr, y_tr)
    cv_results = model.score(X_te,y_true.reshape(-1,1))
    results.append(cv_results)
    names.append(name)
    #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(name, results[i])



for i in ['mse','mae']:
    print("number of estimators: ",i)
    regressor = RandomForestRegressor(n_estimators=100,criterion=i)
    regressor.fit(X_tr, y_tr)
    y_pred = regressor.predict(X_te.reshape(-1,1))
    y_pred = y_pred.reshape(len(y_true),)
    print("Test-set R^2 test: {:.2f}".format(r2_score(y_true, y_pred)))





'''
classifier = LinearRegression()
classifier.fit(X_tr,y_tr)
y_te = classifier.predict(X_te)

average_precision = average_precision_score(y_true,y_te)

disp = plot_precision_recall_curve(classifier, X_te, y_te)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
'''