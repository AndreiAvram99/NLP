
import googletrans as gt
import re
from googletrans import Translator
from sklearn.model_selection import train_test_split

def dataset_info():
    # Print data set info
    print("Dataset size:", len(twitter_train_df), '\n')
    # twitter_train_df.info()

    # Print distribution across all training languages
    tweet_distribution = twitter_train_df.groupby('language').count()['text']

    print(tweet_distribution.head())

twitter_train_df = pd.read_csv('nlp_dataset.csv', on_bad_lines='skip')

twitter_train_df = twitter_train_df.drop('language', axis=1)

translator = Translator()

texts = twitter_train_df['text'].values
labels = twitter_train_df['label'].values

texts = [re.sub(r'@[A-Za-z0-9]+', "", text) for text in texts]

translated_set = []
for idx, text in enumerate(texts):
    if idx%100==0:
        print(idx)
    # translated_text = text
    # if lg[idx].lang != "en":
    translated_text = None
    while translated_text is None:
        try:
            translated_text = translator.translate(text, dest='en').text
        except:
            print("err")
    translated_set.append([translated_text, labels[idx]])

pd.DataFrame(translated_set, columns=["text", "label"]).to_csv("translated_set.csv", index=False)
