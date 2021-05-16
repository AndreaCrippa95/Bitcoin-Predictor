import pandas as pd
import sys
import numpy as np
import math

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/data/DF_train.csv'
df = pd.read_csv(path, header=0)

df.columns.values[0] = 'Date'
df.columns.values[2] = 'Gold'
df.columns.values[3] = 'NASDAQ'

df.head()



#train
model = RandomForestRegressor()
model.fit(df.drop())
