import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

posts = pd.read_csv('tweets.csv')

data = fetch_20newsgroups()
categories = data.target_names

train_data = fetch_20newsgroups(subset='train', categories=categories)

test_data = fetch_20newsgroups(subset='test', categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_data.data, train_data.target)

posts = pd.DataFrame(pd.read_csv('tweets.csv', encoding='latin-1'))
posts.columns = ['1', '2', 'Date', 'Query/No query', 'User', 'Tweet']
posts.head()

df = pd.DataFrame()

def predict_category(train=train_data, model=model, df=df):

    #filt = result == str([x for x in train.target_names])
    for i in range(posts.shape[0]//4000):
        pred = model.predict([posts.loc[i, 'Tweet']])
        #sorted_df.columns = train.target_names
        result = train.target_names[pred[0]]
        df_to_add = {'Category': result, 'Tweet': posts.loc[i, 'Tweet']}
        df = df.append(df_to_add, ignore_index=True)

    return df


df = predict_category()
df.head()
