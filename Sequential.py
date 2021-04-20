import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from datetime import date


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load the data

Ticker = 'btcusd'

start = dt.datetime(2012,1,1)
start = start.strftime("%Y-%m-%d")

end = dt.datetime(2020,1,1)
end = end.strftime("%Y-%m-%d")


df = web.get_data_tiingo(Ticker,start,end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['close'].values.reshape(-1,1))

#Days used to predict future price
days = 60

X_train = []
y_train = []

for i in range(days,len(scaled_data)):
    X_train.append(scaled_data[i-days:i,0])
    y_train.append(scaled_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Build model (units,epochs, batch_size are modifiable)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape = (X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
#Prediction for the next closing price
model.add(Dense(units=1))

model.compile(optimizer='adam', loss= 'mean_squared_error')
model.fit(X_train,y_train,epochs=25,batch_size=32)

#Test model accuracy on existing data

test_start = dt.datetime(2020,1,1)
test_start = test_start.strftime("%Y-%m-%d")

test_end = dt.datetime.now()
test_end = test_end.strftime("%Y-%m-%d")

df_test = web.get_data_tiingo(Ticker,test_start,test_end, api_key = ('eef2cf8be7666328395f2702b5712e533ea072b9'))

actual_prices = df_test['close'].values
tot_df = pd.concat((df['close'],df_test['close']),axis=0)

model_input = tot_df[len(tot_df)-len(df_test)-days:].values
model_input = model_input.reshape(-1,1)
model_input = scaler.transform(model_input)

#Make some predictions on the test data

X_test = []

for i in range(days,len(model_input)):
    X_test.append(model_input[i-days:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#Plot test predictions:

plt.plot(actual_prices, color = 'green', label = 'Actual Price')
plt.plot(predictions, color = 'orange', label = 'Predicted Price')
plt.title('Bitcoin Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


#Predict next day

real_df = [model_input[len(model_input)+1-days:len(model_input+1),0]]
real_df = np.array(real_df)
real_df = np.reshape(real_df,(real_df.shape[0],real_df.shape[1],1))


prediction = model.predict(real_df)
prediction = scaler.inverse_transform(prediction)

print(prediction)



