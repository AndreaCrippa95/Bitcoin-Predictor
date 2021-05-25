from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json , time
import sqlite3
from unidecode import unidecode
from textblob import TextBlob

conn = sqlite3.connect('twitter.db')
c = conn.cursor()

API = 'ih1vfWmIgXNN36iUdtkRA8SNd'
API_secret = 'vZeS0JWBNnaMCLjVVtwEMIFANLTcI5oC1P9N6dpyt4YF4MFPPY'
Access = '103961010-lpTQz31OQJRrm5HNClnJg52JABCbCgDEuglpEMew'
Access_secret = 'YxLpi9IcdnydwfCMXecUYZD3z90kYL5WFD8YAH5rUejWm'

bag_of_words = ['BTC','Bitcoin']


def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))

create_table()

class listener(StreamListener):

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

        except KeyError as e:
            print(str(e))
        return(True)

    def on_error(self, status):
        print (status)

while True:
    try:
        auth = OAuthHandler(API, API_secret)
        auth.set_access_token(Access, Access_secret)
        twitterStream = Stream(auth, listener())
        twitterStream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
