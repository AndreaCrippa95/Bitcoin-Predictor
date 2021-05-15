#Inputs file. Save globally the variables to send to the various files
import datetime as dt
#set start, end times and number of days predicted in the future
start = dt.datetime(2012,1,1)

end = dt.datetime(2021,5,1)

prediction_days = 10

#Choose the input data
BTC_Price = True

Gold_Price = True

NDAQ_Price = True

#Choose a model:
ChModel = 'Sequential'

#Choose the desired output
RES = True

GRA = True

ACC = True
