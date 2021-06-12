import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from Brownian_Motion import Brownian


class Method:

    def __init__(self,df,ChModel,days,Data):
        self.model = ChModel
        self.days = days
        self.df = df
        self.Data = Data

    def MachineLearning(self):
        assert self.model in ['RFR', 'GBR', 'LR','Lasso','KNR','EN','DTR','SVM'], "invalid model"

        if self.model in ['RFR']:
            mod = RandomForestRegressor(n_estimators=50,criterion='mse')
        elif self.model in ['GBR']:
            mod = GradientBoostingRegressor(n_estimators=100)
        elif self.model in ['LR']:
            mod = LinearRegression()
        elif self.model in ['Lasso']:
            mod = Lasso(alpha=1)
        elif self.model in ['KNR']:
            mod = KNeighborsRegressor(n_neighbors=3)
        elif self.model in ['EN']:
            mod = ElasticNet(alpha=1)
        elif self.model in ['DTR']:
            mod = DecisionTreeRegressor(max_depth=5,max_leaf_nodes=15)
        elif self.model in ['SVM']:
            mod = SVR(kernel='rbf', C=1e3, gamma=0.01)

        mod.fit(self.Data.X_tr, self.Data.y_tr.ravel())
        results = mod.predict(self.Data.X_te)
        results = results.reshape(-1, 1)
        results = self.Data.y_scaler.inverse_transform(results)
        return results

    def Sequential(self):
        assert self.model in ['Sequential'], "invalid model"
        #assert self.df.columns < 3, 'Too many variables added for optimal use of this model'

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.df.values.reshape(-1, 1))
        # Days used to predict future price
        days = self.days

        X_train = []
        y_train = []

        for i in range(days, len(scaled_data)):
            X_train.append(scaled_data[i - days:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build model (units,epochs, batch_size are modifiable)

        model = Sequential()

        model.add(LSTM(units=50,activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,activation='relu'))
        model.add(Dropout(0.2))
        # Prediction for the next closing price
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=25, batch_size=32)

        # Predict next day

        prediction = []
        for i in range(days):
            real_df = [scaled_data[len(scaled_data) - (days - i):len(scaled_data)]]
            real_df = np.append(real_df, prediction)
            real_df = np.array(real_df)
            real_df = np.reshape(real_df, (1, days, 1))
            pr = model.predict(real_df)
            prediction.append(pr[0, 0])
            real_df = np.append(real_df, prediction)

        prediction = np.array(prediction)
        prediction = np.reshape(prediction, (1, days, 1))
        results = [scaler.inverse_transform(prediction[i]) for i in range(len(prediction))]
        results = np.array(results)
        return results.reshape(-1, 1)

    def Brownian_Motion(self):
        assert self.model in ['BM'], "invalid model"

        val = self.df[self.df.columns[0]]
        val = val.tail(1)
        a = val.item()
        b = Brownian(s0=a)
        return b.stock_price(deltaT=self.days)

    def DNN(self):
        X_train = np.reshape(self.Data.X_tr, (self.Data.X_tr.shape[0], self.Data.X_tr.shape[1], 1))
        X_test= np.reshape(self.Data.X_te, (self.Data.X_te.shape[0], self.Data.X_te.shape[1], 1))
        y_train = np.reshape(self.Data.y_tr, (self.Data.y_tr.shape[0], self.Data.y_tr.shape[1], 1))

        output_dim = y_train.shape[1]
        model = Sequential()
        model.add(Input(shape=(X_test.shape[1], X_test.shape[2]))) # very important input layer
        model.add(LSTM(64, activation='relu'))
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(output_dim))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=15, batch_size=16)

        results = model.predict(X_test)
        results = self.Data.y_scaler.inverse_transform(results)
        return results
