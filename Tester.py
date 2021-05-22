#Import Data making class, Method caller class and Result maker class
from DataClass import Data
from MethodClass import Method
from ResultClass import Results
#Inputs file. Save globally the variables to send to the various files
import datetime as dt

#set start, end times and number of days predicted in the future
start = dt.datetime(2012,1,1)
end = dt.datetime(2021,5,1)
prediction_days = 10
#Choose the input data
BTC_Price = True
Gold_Price = False
NDAQ_Price = True
#Choose a model:
ChModel = 'DNN'
#Choose the desired output
RES = True
GRA = True
ACC = True

#Launching program
def Launcher():
    dat = Data(start=start, end=end, days=prediction_days, BTC=BTC_Price, Gold=Gold_Price, NDAQ=NDAQ_Price)
    df = dat.create_data()
    met = Method(df, ChModel=ChModel, days=prediction_days)
    if ChModel == 'BM':
        res = met.Brownian_Motion()
    elif ChModel == 'Sequential':
        res = met.Sequential()
    elif ChModel in ['RFR', 'GBR', 'LR', 'Lasso', 'KNR', 'EN', 'DTR']:
        res = met.MachineLearning()
    elif ChModel in ['SVM']:
        res = met.SVM()
    elif ChModel in ['DNN']:
        res = met.DNN()
    elif ChModel in ['RNN']:
        res = met.RNN()

    gmaker = Results(df, res, ChModel=ChModel, end=end, days=prediction_days)
    if GRA:
        gmaker.Graph()
    if RES:
        gmaker.Results()
    if ACC:
        gmaker.Accuracy()
    return

Launcher()