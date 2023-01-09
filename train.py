#import the library
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']
  
    return negative, neutral, positive, compound

twitter_train_df = pd.read_csv('nlp_dataset.csv', on_bad_lines='skip').head(50)

texts = twitter_train_df['text'].values
labels = twitter_train_df['label'].values

train_set_input = []
for idx, text in enumerate(texts):
    train_set_input.append(sentiment_vader(text))

train_set_labels = [(x - 1) / 4 for x in labels]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




