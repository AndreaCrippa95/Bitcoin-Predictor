#Inputs file. Save globally the variables to send to the various files
import os
import sys
path = '/Bitcoin-Predictor/Dashboard/first_try'
sys.path.append(path)
import app_try

import datetime as dt
#set start, end times and number of days predicted in the future
start = app_try.date_string

print(start)
