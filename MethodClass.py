import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from OLD.Brownian_Motion import Brownian

class Method:

    def __init__(self,df,ChModel,days):
        self.model = ChModel
        self.days = days
        self.df = df


    def MachineLearning(self):
        assert self.model in ['RFR', 'GBR', 'LR','Lasso','KNR','EN','DTR'], "invalid model"

        if self.model in ['RFR']:
            mod = RandomForestRegressor()
        elif self.model in ['GBR']:
            mod = GradientBoostingRegressor()
        elif self.model in ['LR']:
            mod = LinearRegression()
        elif self.model in ['Lasso']:
            mod = Lasso()
        elif self.model in ['KNR']:
            mod = KNeighborsRegressor()
        elif self.model in ['EN']:
            mod = ElasticNet()
        elif self.model in ['DTR']:
            mod = DecisionTreeRegressor()

        predictor = self.df['BTC Price'].shift(-self.days)

        X = np.array(self.df)
        X = X[:len(self.df) - self.days]

        y = np.array(predictor)
        y = y[:-self.days]
        y = y.reshape(-1, 1)

        mod.fit(X, y.ravel())
        results = mod.predict(np.array(self.df[-self.days:]))
        return results.reshape(-1, 1)

    def Sequential(self):
        assert self.model in ['Sequential'], "invalid model"
        assert self.df.columns < 2, 'Too many variables added for optimal use of this model'

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

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
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
        #assert self.model in ['BM'], "invalid model"

        val = self.df['BTC Price']
        val = val.tail(1)
        a = val.item()
        b = Brownian(s0=a)
        return b.stock_price(deltaT=self.days)
