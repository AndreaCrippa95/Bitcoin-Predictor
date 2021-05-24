
'''
from twython import Twython
import time
APP_KEY = 'ih1vfWmIgXNN36iUdtkRA8SNd'
APP_SECRET = 'vZeS0JWBNnaMCLjVVtwEMIFANLTcI5oC1P9N6dpyt4YF4MFPPY'
twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
ACCESS_TOKEN = twitter.obtain_access_token()
print(ACCESS_TOKEN)
'''

import time
import pandas as pd
from twython import Twython

APP_KEY = 'ih1vfWmIgXNN36iUdtkRA8SNd'
ACCESS_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAA7mPwEAAAAAYGeHuEro0lG9vIkEO%2B1nOQQoV9w%3DokeFIx0XkZe2YNDt4zxvQpzTyTQiwWLJoivHwR528Z0E5ym0oV'
twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)
twitter.get_application_rate_limit_status()['resources']['search']

search = twitter.search(q='$BTC',count=2000)
tweets = search['statuses']

ids = []
for tweet in tweets:
    ids.append(tweet['id_str'])

ids = [tweet['id_str'] for tweet in tweets]
texts = [tweet['text'] for tweet in tweets]
times = [tweet['retweet_count'] for tweet in tweets]
favtimes = [tweet['favorite_count'] for tweet in tweets]
follower_count = [tweet['user']['followers_count'] for tweet in tweets]
location = [tweet['user']['location'] for tweet in tweets]
lang = [tweet['lang'] for tweet in tweets]

#CORRELATION BETWEEN RETWEET AND FAVORITE COUNTS AND PLOT IT
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
pl = pd.DataFrame(
    {'id': ids,
     'text': texts,
     'retweet_count': times,
     'fav_count':favtimes,
     'follower_count':follower_count,
     'location':location,
     'lang':lang
    })
pl.head(100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pl[['retweet_count','fav_count']].describe().T
pl['retweet_count'].corr(pl['fav_count'])
pl.plot(kind='scatter', x='fav_count', y='retweet_count')
sns.regplot(x='fav_count', y='retweet_count', data=pl)
pl['retweet_count'].plot(kind='line')
plt.show()
sns.show()

import re
def word_in_text(word, text):
 word = word.lower()
 text = text.lower()
 match = re.search(word, text)
 if match:
    return True

pl['ETH'] = pl['text'].apply(lambda tweet: word_in_text('BTC', tweet))
print(pl['ETH'].value_counts()[True])

