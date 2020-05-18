import pandas
import numpy
import nltk
import json
import matplotlib
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path

#json_paths = Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20").glob("**/*.json")
path = Path(r"C:\Users\hyder\Google Drive\Projects\Text Analysis\journey_export_5-11-20\1546305418250-u4ocn4b8hrw1bvw0.json")
df = pandas.DataFrame(columns=['id', 'datetime', 'time', 'year', 'month', 'date', 'day', 'location', 'text', 'att1', 'att2', 'att3'], dtype=object)
json_dicts = []
# Fields:
# Unique ID / Day / Month / Year / Location / Text /
#
#
#


#for path in json_paths:
with open(str(path), mode='r', encoding="utf-8") as file:
    json_full = json.load(file)
    json_text = BeautifulSoup(json_full['text'].replace(u"&nbsp;",' '), features="html.parser").get_text()
    json_datetime = datetime.fromtimestamp(json_full['date_journal']/1000) # consider checking timezone. Or don't if you only care about local time!
    json_time = json_datetime.strftime('%I:%M:%S %p')
    json_date = str(json_datetime.date())
    json_day = json_datetime.strftime('%A')
    json_dict = {'id': json_full['id'], 'datetime': json_datetime, 'time': json_time, 'year': json_date.split('-')[0], 'month': json_date.split('-')[1], 'date': json_date.split('-')[2], 'day': json_day, 'location': [json_full['lat'], json_full['lon']], 'text': json_text}
    print(json_dict)
