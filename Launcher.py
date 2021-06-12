#Import Data making class, Method caller class and Result maker class
from DataClass import Data
from MethodClass import Method
from ResultClass import Results
#Inputs file. Save globally the variables to send to the various files
import datetime as dt

#set start, end times and number of days predicted in the future
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,6,1)
prediction_days = 10
#Choose the input data
BTC_Price = False
Gold_Price = False
NDAQ_Price = False
Returns = True
#Choose a model:
ChModel = ''
#Choose the desired output
RES = True
GRA = True

#Launching program
for ChModel in ['RFR', 'GBR', 'LR','Lasso','KNR','EN','DTR','SVM','Sequential','DNN','BM']:
    dat = Data(start=start,end=end,days=prediction_days,BTC=BTC_Price,Gold=Gold_Price,NDAQ=NDAQ_Price,Returns=Returns)
    dat.create_data()
    df = dat.df
    met = Method(df,ChModel=ChModel,days=prediction_days,Data=dat)
    if ChModel == 'BM':
        res = met.Brownian_Motion()
    elif ChModel == 'Sequential':
        res = met.Sequential()
    elif ChModel in ['RFR', 'GBR', 'LR','Lasso','KNR','EN','DTR','SVM']:
        res = met.MachineLearning()
    elif ChModel in ['DNN']:
        res = met.DNN()
    else:
        raise ValueError

    gmaker = Results(df,res,ChModel=ChModel,end=end,days=prediction_days)
    if GRA:
        gmaker.Graph()
    if RES:
        gmaker.Results()
