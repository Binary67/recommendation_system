from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re

app = FastAPI()

class ScoringItem(BaseModel):
    text: str

def data_cleaning(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = [word for word in text.split() if len(word)>2]
    text = " ".join(text)

    return text

def encode_text(text):

    return vectorizer.transform(list(text))

def make_prediction(encoded_x):

    return model.predict(encoded_x)

model = load('SVM_model.joblib')
vectorizer = load('vectorizer.joblib')
le = preprocessing.LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    item_dict = item.dict()
    df_data = pd.DataFrame(item_dict, index=[0])

    # df_data = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    cleaned_text = df_data['text'].apply(data_cleaning)
    encoded_x = encode_text(cleaned_text)
    
    y_hat = make_prediction(encoded_x.toarray())
    y_label = le.inverse_transform(y_hat)
    
    return {'prediction' : y_label[0]}

