from textblob import TextBlob
import datanews

datanews.api_key = '0ihqel7juf367d6or4qfijfi6'

response = datanews.headlines(q='Bitcoin', language=['en'])
articles = response['hits']

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

