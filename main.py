import pandas as pd
import re
import num2words
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import json
import matplotlib
import matplotlib.pyplot as plt
import string
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
import gensim
from gensim.models import Word2Vec
import variables

json_paths = Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20").glob("**/*.json")
#path = Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20\1546305418250-u4ocn4b8hrw1bvw0.json")
json_dicts = []
full_text_all = ""
sent_tokens_all = []
word_tokens_all = []

stop_words=set(stopwords.words("english") + list(string.punctuation) + variables.stop_words_additional)
contractions = re.compile('({keys})'.format(keys = '|'.join(variables.cList.keys())), re.IGNORECASE)

def str_escape(str, encoding="utf-8"):
    return (str.encode('latin1').decode('unicode-escape').encode('latin1').decode(encoding))

def pretty_pd_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n')
        print("Dataframe")
        print(df)

def text_replacements(str): # TO DO: Use regex to replace any occurences of char followed by \n with '. '. And also replace &nbsp; with ' '

    str = str.replace(u"&nbsp;",' ')

    # TO DO: Check using r character!
    to_replace = (r'\.{3}', r'(\w)\n', r'([.?!, ])\1+') # replace '...' with ' ', replace end of sentence without period + newline with '. ', remove duplicate punctuations
    replace_with = (r' ', r'\1. ', r'\1')
    printable = set(string.printable)

    for pair in zip(to_replace, replace_with):
        str=re.sub(pair[0], pair[1], str)

    str = re.sub(r'(\d+)', lambda x: num2words.num2words(int(x.group(0))), str)
    str = ''.join(filter(lambda x: x in printable, str))
    return str

def expand_contractions(str):
    def replace(contraction):
        contraction_str = contraction.group(0).lower()
        expansion = variables.cList[contraction_str]
        # if contraction_str.islower(): expansion = expansion.lower()
        # if contraction_str.isupper(): expansion = expansion.upper()
        # if contraction_str.istitle(): expansion = expansion.title()
        return expansion
    str = contractions.sub(replace, str)
    return str

def text_process(raw_text):

    full_text = BeautifulSoup(raw_text, features="html.parser").get_text().lower() # Making everything lowercase is easier for now
    full_text = text_replacements(full_text)
    full_text = expand_contractions(full_text)

    sent_tokens = nltk.sent_tokenize(full_text)
    word_tokens = []
    for word in nltk.word_tokenize(full_text):
        if word not in stop_words:
            word_tokens.append(word)

    return full_text, sent_tokens, word_tokens

for idx,path in enumerate(json_paths):
    with open(str(path), mode='r', encoding="utf-8") as file:
        json_full = json.load(file)

        full_text, sent_tokens, word_tokens = text_process(json_full['text'])
        json_datetime = datetime.fromtimestamp(json_full['date_journal']/1000) # consider checking timezone. Or don't if you only care about local time!
        json_time = json_datetime.strftime('%I:%M:%S %p')
        json_date = str(json_datetime.date())
        json_day_of_week = json_datetime.strftime('%A')
        json_lat = (json_full['lat'],None)[json_full['lat']>900] # Handle invalid lat/longs
        json_lon = (json_full['lon'],None)[json_full['lon']>900]

        json_dict = {'id': json_full['id'], 'date_time': json_datetime, 'date': json_date, 'time': json_time, 'year': json_date.split('-')[0], 'month': json_date.split('-')[1], 'day': json_date.split('-')[2], 'day_of_week': json_day_of_week, 'location': [json_lat, json_lon], 'full_text': full_text, 'sent_tokens': sent_tokens, 'word_tokens': word_tokens}

        full_text_all = full_text_all + '. ' + full_text
        sent_tokens_all = sent_tokens_all + sent_tokens
        word_tokens_all = word_tokens_all + word_tokens

        json_dicts.append(json_dict)

full_df = pd.DataFrame.from_dict(json_dicts)

#print(full_text)
pretty_pd_print(full_df)

print("Number of days: ")
print(full_df['date'].nunique())

freq_dist_word = FreqDist(word_tokens_all)
freq_dist_word.plot(100,cumulative=False)
