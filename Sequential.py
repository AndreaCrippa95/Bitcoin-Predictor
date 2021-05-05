#Use the Sequential Neural Network to predict the price of the Bitcoin
#Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Load the data from Database.py
prediction_days = 60

df = pd.read_csv('data/DataFrame',index_col=0)
df.index = df.index.astype('<M8[ns]')
#Scaling the data to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values.reshape(-1,1))
model = 'Sequential Neural Network'
#Days used to predict future price
days = prediction_days

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

#Predict next day

prediction = []
real_df = [scaled_data[len(scaled_data)-(days):len(scaled_data)]]
real_df = np.array(real_df)
real_df = np.append(real_df, prediction)
real_df = np.reshape(real_df,(1,days,1))


prediction = model.predict(real_df)
prediction = scaler.inverse_transform(prediction)

print(prediction)

prediction = []
for i in range(days):
    real_df = [scaled_data[len(scaled_data)-(days-i):len(scaled_data)]]
    real_df = np.append(real_df, prediction)
    real_df = np.array(real_df)
    real_df = np.reshape(real_df, (1, days, 1))
    pr = model.predict(real_df)
    prediction.append(pr[0,0])
    real_df = np.append(real_df,prediction)

prediction = np.array(prediction)
prediction = np.reshape(prediction,(1,days,1))
results = [scaler.inverse_transform(prediction[i]) for i in range(len(prediction))]
results = np.array(results)
results = results.reshape(-1,1)
np.savetxt('data/results',results)