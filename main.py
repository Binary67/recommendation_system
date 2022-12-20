import pandas as pd
import numpy as np
import string
import re
import spacy
from collections import Counter

df_data = pd.read_csv('input/df_main.csv')
df_data = df_data[['title', 'category']]
df_data = df_data.sample(100)

NLP = spacy.load('en_core_web_lg')

# Data Cleaning
def data_cleaning(corpus, label = False):
    corpus = corpus.lower()
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    corpus = re.sub(' {2,}', ' ', corpus)
    corpus = re.sub('^\s', '', corpus)
    corpus = re.sub('\s$', '', corpus)

    if label == False:
        corpus = ' '.join(token.lemma_ for token in NLP(corpus) if not token.is_stop)

    return corpus

df_data['title'] = df_data['title'].apply(data_cleaning)
df_data['category'] = df_data.apply(lambda x: data_cleaning(x['category'], True), axis=1)

