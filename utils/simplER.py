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

# -------------------------------
# Ner functions
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