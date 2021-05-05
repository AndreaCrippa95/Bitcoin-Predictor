#Inputs file. Save globally the variables to send to the various files
import datetime as dt

def Init():
    global start
    start = dt.datetime(2018,1,1)
    global end
    end  = dt.datetime(2019,1,1)
    global prediction_days
    prediction_days = 60