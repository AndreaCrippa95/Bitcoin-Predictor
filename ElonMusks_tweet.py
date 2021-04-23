import glob
import re
import pandas as pd
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

CV = CountVectorizer( )
x = CV.fit_transform(df['tweet'])

x = x.toarray()
total = x.sum(axis=1)
total = pd.Series(total)
max_location = total.nlargest(n=50)

# create a function to clean the tweets
def cleanTwt(twt):
    twt = re.sub("#bitcoin", 'bitcoin', twt) # removes the '#' from bitcoin
    twt = re.sub("#Bitcoin", 'Bitcoin', twt) # removes the '#' from Bitcoin
    twt = re.sub('#[A-Za-z0-9]+', '', twt) # removes any string with a '#'
    twt = re.sub('\\n', '', twt) # removes the '\n' string
    twt = re.sub('https:\/\/\S+', '', twt) # removes any hyperlinks
    return twt

df['cleaned_tweets'] = df['tweet'].apply(cleanTwt)

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

plt.figure(figsize=(14,10))

for i in range(0, 1000):
    plt.scatter(df["polarity"].iloc[[i]].values[0], df["subjectivity"].iloc[[i]].values[0], color="Purple")

plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()
