from __future__ import division
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.classify.scikitlearn import SklearnClassifier

import math
import random
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import requests
import sys
import sqlite3
import pprint
import pandas as pd
from pandas.io.json import json_normalize
#encoding=utf8

reload(sys)
sys.setdefaultencoding('utf8')


def majority( max_accuracy_list ):
    max_accuracy_array = np.array(max_accuracy_list)
    index = np.argmax(max_accuracy_array)
    return index

def classification(feature_sets):
    random.shuffle(feature_sets)
    bayes_runs = []
    num_folds_bayes = 10
    subset_size_bayes = int(math.ceil(len(feature_sets) / num_folds_bayes))

    for i in range(num_folds_bayes):
        training_set_bayes = (feature_sets[(i + 1) * subset_size_bayes:] + feature_sets[:i * subset_size_bayes])
        testing_set_bayes = feature_sets[i * subset_size_bayes:(i + 1) * subset_size_bayes]
    global classifier_bayes
	classifier_bayes = nltk.NaiveBayesClassifier.train(training_set_bayes)
    bayes_runs.append((nltk.classify.accuracy(classifier_bayes, testing_set_bayes)) * 100)

    random.shuffle(feature_sets)
    MNB_1 = []
    BR_2 = []
    LR_3 = []
    SGD_4 = []
    SVC_5 = []
    RFC_6 = []
    num_folds_7 = 10
    subset_size_7 = int(math.ceil(len(feature_sets) / num_folds_7))
    global mnb
    global br
    global lr
    global sgd
    global svc
    global rfc


    for i in range(num_folds_7):
        training_set_7 = (feature_sets[(i + 1) * subset_size_7:] + feature_sets[:i * subset_size_7])
        testing_set_7 = feature_sets[i * subset_size_7:(i + 1) * subset_size_7]

    MNB_classifier_7 = SklearnClassifier(MultinomialNB())
	mnb = MNB_classifier_7.train(training_set_7)
    MNB_1.append((nltk.classify.accuracy(MNB_classifier_7, testing_set_7)) * 100)
    BernoulliNB_Classifier_7 = SklearnClassifier(BernoulliNB())
    br=BernoulliNB_Classifier_7.train(training_set_7)
    BR_2.append((nltk.classify.accuracy(BernoulliNB_Classifier_7, testing_set_7)) * 100)
    LogisticRegression_Classifier_7 = SklearnClassifier(LogisticRegression())
	lr=LogisticRegression_Classifier_7.train(training_set_7)
    LR_3.append((nltk.classify.accuracy(LogisticRegression_Classifier_7, testing_set_7)) * 100)
    SGD_Classifier_7 = SklearnClassifier(SGDClassifier())
    sgd = SGD_Classifier_7.train(training_set_7)
    SGD_4.append((nltk.classify.accuracy(SGD_Classifier_7, testing_set_7)) * 100)
    LinearSVC_Classifier_7 = SklearnClassifier(LinearSVC())
    svc = LinearSVC_Classifier_7.train(training_set_7)
    SVC_5.append((nltk.classify.accuracy(LinearSVC_Classifier_7, testing_set_7)) * 100)
    RFClassifier_Classifier_7 = SklearnClassifier(RandomForestClassifier())
    rfc = RFClassifier_Classifier_7.train(training_set_7)
    RFC_6.append(nltk.classify.accuracy(RFClassifier_Classifier_7, testing_set_7)*100)

    max_accuracy_list = [max(bayes_runs) , max(MNB_1) , max(BR_2) , max(LR_3) , max(SGD_4) , max(SVC_5) , max(RFC_6)]
    max_accuracy_model = {max(bayes_runs):classifier_bayes,max(MNB_1):mnb}
    global max_accuracy_classifier 
    max_accuracy_classifier = majority(max_accuracy_list)
    if  max_accuracy_classifier == 0:
	    return classifier_bayes
    elif max_accuracy_classifier ==1:
	    return mnb
    elif max_accuracy_classifier == 2:  
        return br	
    elif max_accuracy_classifier == 3:  
        return lr	
    elif max_accuracy_classifier == 4:  
        return sgd
    elif max_accuracy_classifier == 5:  
        return svc
    elif max_accuracy_classifier == 6:  
        return rfc

def updateCoinTable():
	response = requests.get("https://min-api.cryptocompare.com/data/pricemulti?fsyms=BTC,ETH,BCH,LTC,XRP&tsyms=USD,CAD,CNY,KRW,INR,JPY,EUR,AUD,RUB")
	data=json.loads(response.text)

	df=json_normalize(data)
	print("coins data--->",df)
	coins=['LTC','ETH','BCH','XRP','BTC']
	currency=['USD','AUD','RUB','KRW','INR','CNY','JPY','EUR','CAD']

	conn=sqlite3.connect('dashboard.db')
	c=conn.cursor()
	update_query='UPDATE Coin SET USD=?, AUD=?, RUB=?, KRW=?, INR=?, CNY=?, JPY=?, EUR=?, CAD=? WHERE Coin_Name=?'

	for i in coins:
		currentcoinName= i
		USD=data[i]['USD']
		AUD=data[i]['AUD']
		RUB=data[i]['RUB']
		KRW=data[i]['KRW']	
		INR=data[i]['INR']
		CNY=data[i]['CNY']
		JPY=data[i]['JPY']
		EUR=data[i]['EUR']
		CAD=data[i]['CAD']
		c.execute(update_query, (USD, AUD, RUB, KRW, INR, CNY, JPY, EUR, CAD, currentcoinName))

	conn.commit()
	conn.close()

def updateICO():
	utcMap = {"UTC+0":"GBR","UTC+3":"RUS","UTC--8":"USA","UTC+1":"DEU","UTC--4":"BRA","UTC--7":"CAN","UTC--5":"PER","UTC+7":"THA"}
	response = requests.get("https://api.icowatchlist.com/public/v1/upcoming")
	data=json.loads(response.text)
	icos= data["ico"]["upcoming"]

	conn=sqlite3.connect('dashboard.db')
	c=conn.cursor()
	insert_query='INSERT into ICO VALUES(?,?,?,?,?,?,?,?,?,?)'

	c.execute('Select max(Obj_Id) from ICO')
	max_objid= c.fetchone()
	objidRange=0;
	if max_objid[0] > 0:
		objidRange = max_objid[0]
		objidRange = objidRange+1

	for i in range(0,len(icos)):
		obj_id=objidRange
		print(obj_id)
		name=icos[i]['name']
		image=icos[i]['image']
		description=icos[i]['description']
		website_link=icos[i]['website_link']
		icowatchlist_url=icos[i]['icowatchlist_url']
		start_time=icos[i]['start_time']
		end_time=icos[i]['end_time']
		timezone=icos[i]['timezone']
		country = utcMap[timezone]
		c.execute(insert_query, (obj_id,name,image,description,website_link,icowatchlist_url,start_time,end_time,timezone,country))
		objidRange = objidRange+1

	conn.commit()
	conn.close()
	
def createDb():
	conn = sqlite3.connect('dashboard.db')
	c = conn.cursor()  #get a cursor object, all SQL commands are processed by it
	c.execute("CREATE TABLE ICO (Obj_Id INTEGER,ICO_Name TEXT,image TEXT,description TEXT,website_link TEXT,icowatchlist_url TEXT,start_time NUMERIC,end_time NUMERIC,timezone TEXT,country TEXT, PRIMARY KEY(Obj_Id))") #create a table
	c.execute("CREATE TABLE Coin (Obj_Id INTEGER,Coin_Name TEXT,USD NUMERIC,AUD NUMERIC,RUB NUMERIC,KRW NUMERIC,INR NUMERIC,CNY NUMERIC,JPY NUMERIC,EUR NUMERIC,CAD NUMERIC,sentiment NUMERIC,PRIMARY KEY(Obj_Id))") #create a table
	c.execute("INSERT INTO Coin VALUES(1,'BTC',0,0,0,0,0,0,0,0,0,0)")
	c.execute("INSERT INTO Coin VALUES(2,'BCH',0,0,0,0,0,0,0,0,0,0)")
	c.execute("INSERT INTO Coin VALUES(3,'ETH',0,0,0,0,0,0,0,0,0,0)")
	c.execute("INSERT INTO Coin VALUES(4,'XRP',0,0,0,0,0,0,0,0,0,0)")
	c.execute("INSERT INTO Coin VALUES(5,'LTC',0,0,0,0,0,0,0,0,0,0)")

	conn.commit() #save the changes
	conn.close()

con = sqlite3.connect('dashboard.db')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tableList = cursor.fetchall()
if len(tableList) < 1:
	createDb()
con.close()
updateCoinTable()
updateICO()
performanceBCH()
performanceBTC()
performanceETH()
performanceLTC()
performanceXRP()
updateSentimentInDB(sentimentDict)








