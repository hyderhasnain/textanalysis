import pandas as pd
import re
import num2words
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import matplotlib
import matplotlib.pyplot as plt
import statistics as stats
import string
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path
import gensim
from gensim.models import Word2Vec
import variables

json_paths = Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20").glob("**/*.json")
#json_paths = [Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20\1546305418250-u4ocn4b8hrw1bvw0.json")]
json_dicts = []
full_text_all = ""
sent_tokens_all = []
word_tokens_all = []
extra_tokens_all = []

stop_words=set(stopwords.words("english") + list(string.punctuation) + variables.stop_words_additional)
contractions = re.compile('({keys})'.format(keys = '|'.join(variables.cList.keys())), re.IGNORECASE)

lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()
new_sentiment_words = {'kill': -5, 'hate': -2}
sentiment_analyzer.lexicon.update(new_sentiment_words)

def str_escape(str, encoding="utf-8"):
    return (str.encode('latin1').decode('unicode-escape').encode('latin1').decode(encoding))

def pretty_pd_print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2500):
        print('\n')
        print("Dataframe")
        print(df[['id', 'date', 'full_text', 'com_sentiment']])

def text_replacements(str): # TO DO: Use regex to replace any occurences of char followed by \n with '. '. And also replace &nbsp; with ' '

    printable = set(string.printable)
    str = re.sub(r'\b(?!(?:\d{4}\b))\d+', lambda x: ' ' + num2words.num2words(int(x.group(0))) + ' ', str)
    str = ''.join(filter(lambda x: x in printable, str))

    # replace non-breaking space HTML tags, replace ... with ' ', reaplce newline following a char with . <space> following that char, remove duplicate puncutations, remove '-ish' suffixes
    to_replace = (r'&nbsp;', r'\.{3}', r'(\w)\n', r'([.?!, \n\"`])\1+', r'\b(\w+)(\-ish)\b')
    replace_with = (' ', r' ', r'\1. ', r'\1', r'\1')

    for pair in zip(to_replace, replace_with):
        str=re.sub(pair[0], pair[1], str)

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

    full_sentiments = {'pos': [], 'neg': [], 'neu': [], 'com': [], 'count': 0}

    full_text = BeautifulSoup(raw_text, features="html.parser").get_text().lower() # Making everything lowercase is easier for now
    full_text = text_replacements(full_text)
    full_text = expand_contractions(full_text)

    if full_text:  # Only do more processing if our full_text has any contents, otherwise further processing will error

        word_tokens = []
        sent_tokens = nltk.sent_tokenize(full_text) # tokenize each sentence

        for sent in sent_tokens:

            # Perform sentiment analysis on each sentence
            sent_sentiments = sentiment_analyzer.polarity_scores(sent)
            full_sentiments['pos'].append(sent_sentiments['pos'])
            full_sentiments['neg'].append(sent_sentiments['neg'])
            full_sentiments['neu'].append(sent_sentiments['neu'])
            full_sentiments['com'].append(sent_sentiments['compound'])

            temp_tokens = nltk.word_tokenize(sent) # tokenized words in each sentence

            # Perform word tokenization, stop word removal, and lemmatization
            for word, tag in pos_tag(temp_tokens):
                if tag.startswith('NN'):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'

                word = lemmatizer.lemmatize(word, pos)

                if word not in stop_words: word_tokens.append(word)

        #sentiment_scores = {'pos': sum(full_sentiments['pos'])/len(full_sentiments['pos']), 'neg': sum(full_sentiments['neg'])/len(full_sentiments['neg']), 'neu': sum(full_sentiments['neu'])/len(full_sentiments['neu']), 'com': sum(full_sentiments['com'])/len(full_sentiments['com'])}
        sentiment_scores = {'pos': stats.mean(full_sentiments['pos']), 'neg': stats.mean(full_sentiments['neg']), 'neu': stats.mean(full_sentiments['neu']), 'com': stats.mean(full_sentiments['com'])}

    else:
        sent_tokens = ""
        word_tokens = ""
        sentiment_scores = ""

    return full_text, sent_tokens, word_tokens, sentiment_scores

for idx,path in enumerate(json_paths):
    with open(str(path), mode='r', encoding="utf-8") as file:
        json_full = json.load(file)

        full_text, sent_tokens, word_tokens, sentiment_scores = text_process(json_full['text'])
        if not full_text: continue # In case this entry had no usable text
        json_datetime = datetime.fromtimestamp(json_full['date_journal']/1000) # consider checking timezone. Or don't if you only care about local time!
        json_time = json_datetime.strftime('%I:%M:%S %p')
        json_date = str(json_datetime.date())
        json_day_of_week = json_datetime.strftime('%A')
        json_lat = (json_full['lat'],None)[json_full['lat']>900] # Handle invalid lat/longs
        json_lon = (json_full['lon'],None)[json_full['lon']>900]

        json_dict = {'id': json_full['id'], 'date_time': json_datetime, 'date': json_date, 'time': json_time, 'year': json_date.split('-')[0], 'month': json_date.split('-')[1], 'day': json_date.split('-')[2], 'day_of_week': json_day_of_week, 'location': [json_lat, json_lon], 'full_text': full_text, 'sent_tokens': sent_tokens, 'word_tokens': word_tokens, 'pos_sentiment': sentiment_scores['pos'], 'neg_sentiment': sentiment_scores['neg'], 'neu_sentiment': sentiment_scores['neu'], 'com_sentiment': sentiment_scores['com']}

        full_text_all = full_text_all + '. ' + full_text
        sent_tokens_all = sent_tokens_all + sent_tokens
        word_tokens_all = word_tokens_all + word_tokens

        json_dicts.append(json_dict)

full_df = pd.DataFrame.from_dict(json_dicts)

print(full_text)
pretty_pd_print(full_df)

print("Number of days: ")
print(full_df['date'].nunique())

freq_dist_word = FreqDist(word_tokens_all)
print(freq_dist_word.most_common(30))
#freq_dist_word.plot(100,cumulative=False)
