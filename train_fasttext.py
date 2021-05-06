import csv
import pandas as pd
from prediction_models.services.pre_process_text import remove_punctuations
from prediction_models.services.pre_process_text import replace_special_chars
from prediction_models.services.pre_process_text import remove_stop_words
from prediction_models.services.pre_process_text import rejoin_text
from fasttext import FastText
import pickle


rating = {'1':'Very_Bad', '2':'Bad','3':'Neutral','4':'Good','5':'Very_Good'}
#pre-process review column
df = pd.read_csv('dataset.csv',encoding='utf-8-sig', dtype = str)
df['review'] = remove_punctuations(df['review'] )
df['review'] = df['review'].apply(replace_special_chars)
df['review'] = df['review'].apply(remove_stop_words)
df['review'] = df['review'].apply(rejoin_text)

df.to_csv('dataset.csv',encoding='utf-8-sig')

df.head()