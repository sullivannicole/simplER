# ------------------------
# Get Spark session
# ------------------------

from pyspark.sql import SparkSession
spark = SparkSession.getActiveSession()

# ------------------------
# Imports
# ------------------------

import pandas as pd
from flair.nn import Classifier
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from bs4 import BeautifulSoup
import requests
import re
import time
import numpy as np

# -------------------------------
# NER functions
# -------------------------------

def split_sentences(text):

  split_list = []

  for i, ele in enumerate(text):

    # use splitter to split text into list of sentences
    splitter = SegtokSentenceSplitter()
    sentences = splitter.split(ele)

    split_list.append({'id': i, 'input_text': ele, 'sentences': sentences})

  split_df = pd.DataFrame(split_list)

  return split_df

def run_NER(df):
  '''
  Attributes:
    df: pandas DataFrame with a unique identifier column (id), and a list-column with Flair sentences corresponding to the id (specifically designed to take the output of split_sentences)
  '''
  ner_results = []

  for index, row in df.iterrows():

    sentences = row['sentences']
    
    # predict tags for sentences
    tagger.predict(sentences)

    for sentence in sentences:
      sentence_results = [{'id': row['id'], 'text': x.data_point.text, 'entity_detected': x.value, 'score': x.score} for x in sentence.get_labels('ner')]

      if sentence_results:
        ner_results += sentence_results

  return pd.DataFrame(ner_results)

# ---------------------------------
# Doc retrieval 'graph' function
# ---------------------------------

def run_search(search_query: str, num_results: int = 5):

  time.sleep(0.25)
  # Rotate headers
  headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:84.0) Gecko/20100101 Firefox/84.0'}
  params = {'q': search_query, 'kl': 'us-en'}
  
  page = requests.get('https://html.duckduckgo.com/html', headers = headers, params = params).text
  soup = BeautifulSoup(page, 'html.parser').find_all("a", class_ = "result__url", href = True, limit = num_results)

  urls = [re.sub('\n', '', x.contents[0]).strip() for x in soup] # get URLs
  urls_split = [x.split('/')[1:] for x in urls] # Pull out every after first backslash
  url_titles = [re.sub('-|_', ' ', ' '.join(x)).strip() for x in urls_split] # Concatenate strings into one search term

  return {'search_term': search_query, 'url_title': url_titles}