#Creating a database for the project
#Imports:
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Download data:
df = pd.read_csv('Project/BTC_price.csv')

#Clean the data:

df = df.set_index('Timestamp')
df = df.set_index(pd.to_datetime(df.index))
df = df.rename(columns={"market-price": "Price"})

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
