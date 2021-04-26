import glob
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

path = r'/Users/flavio/Documents/GitHub/Bitcoin-Predictor/data/EM_tweet/' # use your path
all_files = glob.glob(path + "/*.csv")

all = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    all.append(df)

df = pd.concat(all, axis=0, ignore_index=True)


list(df.columns.values)
final_table_columns = ['id', 'date', 'tweet', 'reply_to']

df = df[df.columns.intersection(final_table_columns)]
df.sort_values(by=['date'], ascending=True)

df.head()
CV = CountVectorizer( )
x = CV.fit_transform(df['tweet'])

x = x.toarray()
total = x.sum(axis=1)
total = pd.Series(total)
max_location = total.nlargest(n=50)

# create a function to clean the tweets
def cleanTwt(twt):
    twt = re.sub("#bitcoin", 'bitcoin', twt) # removes the '#' from bitcoin
    twt = re.sub("#Bitcoin", 'bitcoin', twt) # removes the '#' from Bitcoin
    twt = re.sub("#BTC", 'bitcoin', twt)
    twt = re.sub("BTC", 'bitcoin', twt)
    twt = re.sub("$BTC", 'bitcoin', twt)
    twt = re.sub("$btc", 'bitcoin', twt)
    twt = re.sub(r'https?:\/\/.*\/\w*',' ',twt)
    return twt


mylist = ['BTC', 'btc', '$BTC', '$btc', 'bitcoin', 'BITCOIN', 'crypto', 'CRYPTO', '$Doge', 'Doge', 'DOGE', 'dogecoin', '$dogecoin']
pattern = '|'.join([f'(?i){a}' for a in mylist])

df['cleaned_tweets'] = df['tweet'].apply(cleanTwt)

df['cleaned_tweets_2'] = df.cleaned_tweets.str.split(expand=False)

df['Cryto Tweets'] = df['cleaned_tweets_2'].str.contains(pattern)
df.head()

df['Crypto'] = np.where(df['Cryto Tweets'] == 'True', 1, 0)
df_new = df.drop(df[df['Crypto']==0].index, inplace=True)
df_new.head()

df[['Cryto Tweets']].sample(100, random_state=42)

#Part for sentiment analysis
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

# create two new columns called "Subjectivity" & "Polarity"
df['subjectivity'] = df['cleaned_tweets'].apply(getSubjectivity)
df['polarity'] = df['cleaned_tweets'].apply(getPolarity)

def getSentiment(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['polarity'].apply(getSentiment)

"""
plt.figure(figsize=(14,10))

for i in range(0, 1000):
    plt.scatter(df["polarity"].iloc[[i]].values[0], df["subjectivity"].iloc[[i]].values[0], color="Purple")

plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()
plt.savefig("Sentiment Analysis Scatter Plot.png")
"""

#searchfor = ['BTC', 'btc', '$BTC', '$btc', 'bitcoin', 'BITCOIN', 'crypto', 'CRYPTO', '$Doge', 'Doge', 'DOGE', 'dogecoin', '$dogecoin']
