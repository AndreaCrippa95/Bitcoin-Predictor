#Creating a program that creates various predictions with different models
#Imports:
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from Database import df, prediction_days

'''
df['Prediction'] = df[[0]].shift(-prediction_days)

#CREATE THE INDEPENDENT DATA SET (X)

# Convert the dataframe to a numpy array and drop the prediction column
X = np.array(df.drop(['Prediction'],1))

#Remove the last 'n' rows where 'n' is the prediction_days
X= X[:len(df)-prediction_days]


#CREATE THE DEPENDENT DATA SET (y)
# Convert the dataframe to a numpy array (All of the values including the NaN's)
y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = y[:-prediction_days]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
    kfold = KFold(n_splits=10, random_state=0,shuffle=True)
    model.fit(X_train, y_train)
    cv_results = model.score(X_test, y_test)
    results.append(cv_results)
    names.append(name)
    #msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(name, results[i])
'''

#Scaler missing...
df = df.dropna()

predictor = df['BTC Price'].shift(-prediction_days)

X = np.array(df)
X = X[:len(df)-prediction_days]

y = np.array(predictor)
y = y[:-prediction_days]
y = y.reshape(-1,1)

#model = RandomForestRegressor()

#model = GradientBoostingRegressor()

model = LinearRegression()

#model = GradientBoostingRegressor()

#model = RandomForestRegressor()

#model = Lasso()

#model = KNeighborsRegressor()

#model = ElasticNet()

#model = DecisionTreeRegressor()

model.fit(X, y.ravel())
results = model.predict(np.array(df[-prediction_days:]))
results = results.reshape(-1,1)

