from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json, time
import sqlite3
from unidecode import unidecode
from textblob import TextBlob

#Creating DB and Tables to store Stream Data
conn = sqlite3.connect('twitter.db')
c = conn.cursor()

import sys
import os
path = '/Users/flavio/Documents/GitHub/Bitcoin-Predictor/Dashboard/Static/admin'
sys.path.append(path)
import references as ref

API = ref.API
API_secret = ref.API_secret
Access = ref.Access
Access_secret = ref.Access_secret

def table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except:
        pass

table()

class twitter_listener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            analysis = TextBlob(tweet)
            sentiment = analysis.sentiment.polarity
            print(time_ms, tweet, sentiment)
            c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
                      (time_ms, tweet, sentiment))
            conn.commit()

        except:
            pass
        return(True)

while True:
    try:
        auth = OAuthHandler(API, API_secret)
        auth.set_access_token(Access, Access_secret)
        twitterStream = Stream(auth, twitter_listener())
        twitterStream.filter(track=['a','e','i','o','u'])
    except:
        pass
        time.sleep(5)
