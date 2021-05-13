import pandas as pd
import numpy as np
import Inputs
import datetime as dt

start = Inputs.start
end = Inputs.end
prediction_days = Inputs.prediction_days
RES = Inputs.RES

if RES:
    s = end
    e = end + dt.timedelta(prediction_days-1)
    a = start.strftime("%m/%d/%Y")
    b = s.strftime("%m/%d/%Y")
    c = e.strftime("%m/%d/%Y")
    date = pd.date_range(start=b, end=c)
    df2 = pd.DataFrame(index=date)

    results = pd.read_csv('data/results',header=None)
    results = np.array(results)
    prepred = pd.DataFrame(results, index=date)
    pred = pd.merge(df2,prepred, how='outer', left_index=True, right_index=True)
    pred.rename(columns ={0:'Prediction'}, inplace = True)

    print('\nResults:', file=open('data/Result.txt', 'w'))
    print('{:<10}{:>13}'.format('Date', 'BTC Price'), file=open('data/Result.txt', 'a'))
    print('-' * 80, file=open('data/Result.txt', 'a'))
    print(pred['Prediction'], file=open('data/Result.txt', 'a'))
    print('-' * 80, file=open('data/Result.txt', 'a'))
