# Databricks notebook source
!pip install flair
import pandas as pd
from flair.nn import Classifier
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter

# -------------
# Load model
# -------------

tagger = Classifier.load('ner-ontonotes-large')

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


# ---------
# Data
# ---------

imdb_reviews = pd.read_csv('https://raw.githubusercontent.com/sullivannicole/simplER/main/data/imdb_reviews.csv')
imdb_reviews = imdb_reviews.rename(columns={imdb_reviews.columns[0]: 'id'})

# spark.createDataFrame(imdb_reviews).write.mode('overwrite').saveAsTable('user_nsulliv3.simplER_imdb')

rvw_text = imdb_reviews.user_review.values
split_df = split_sentences(rvw_text)
ner_results = run_NER(split_df)

spark.createDataFrame(ner_results).write.mode('overwrite').saveAsTable('user_nsulliv3.simplER_imdb_NER')

# COMMAND ----------

# -------------
# Evaluation
# -------------

# Majority voting
majority_votes = spark.sql('''
WITH art_ents AS (SELECT id,  REGEXP_REPLACE(REGEXP_REPLACE(text, "'", ''), '"|\\.|\\)|\\`', '') AS text
FROM user_nsulliv3.simplER_imdb_NER
WHERE entity_detected = 'WORK_OF_ART'),

token_counts AS (SELECT id, text, COUNT(*) AS n_occur
FROM art_ents
GROUP BY 1, 2),

max_token_counts AS (SELECT *, MAX(n_occur) OVER (PARTITION BY id) AS max_occur
FROM token_counts)

SELECT a.id, a.text AS title_predicted, b.movie
FROM max_token_counts a
LEFT JOIN user_nsulliv3.simplER_imdb b
ON a.id = b.id
WHERE a.n_occur = a.max_occur
ORDER BY id;''')

# Exact match w/ majority voting
spark.sql('''
WITH art_ents AS (SELECT id,  REGEXP_REPLACE(REGEXP_REPLACE(text, "'", ''), '"|\\.|\\)|\\`', '') AS text
FROM user_nsulliv3.simplER_imdb_NER
WHERE entity_detected = 'WORK_OF_ART'),

token_counts AS (SELECT id, text, COUNT(*) AS n_occur
FROM art_ents
GROUP BY 1, 2),

max_token_counts AS (SELECT *, MAX(n_occur) OVER (PARTITION BY id) AS max_occur
FROM token_counts)

SELECT COUNT(DISTINCT a.id) AS n_exact_matches
FROM max_token_counts a
LEFT JOIN user_nsulliv3.simplER_imdb b
ON a.id = b.id
WHERE a.n_occur = a.max_occur
-- Need to strip out punctuation from ground-truth since we also stripped preds of punctuation
AND LOWER(a.text) = LOWER(REGEXP_REPLACE(REGEXP_REPLACE(b.movie, "'", ''), '"|\\.|\\)|\\`', ''))
;''')

# Majority votes for movies without an **exact** match - might have a "containment" match: 180
spark.sql('''
WITH art_ents AS (SELECT id,  REGEXP_REPLACE(REGEXP_REPLACE(text, "'", ''), '"|\\.|\\)|\\`', '') AS text
FROM user_nsulliv3.simplER_imdb_NER
WHERE entity_detected = 'WORK_OF_ART'),

token_counts AS (SELECT id, text, COUNT(*) AS n_occur
FROM art_ents
GROUP BY 1, 2),

max_token_counts AS (SELECT *, MAX(n_occur) OVER (PARTITION BY id) AS max_occur
FROM token_counts),

ids_with_match (SELECT DISTINCT a.id
FROM max_token_counts a
LEFT JOIN user_nsulliv3.simplER_imdb b
ON a.id = b.id
WHERE a.n_occur = a.max_occur
AND LOWER(a.text) = LOWER(REGEXP_REPLACE(REGEXP_REPLACE(b.movie, "'", ''), '"|\\.|\\)|\\`', ''))),

entities_overlap AS (SELECT a.*, b.movie, CONTAINS(LOWER(REGEXP_REPLACE(REGEXP_REPLACE(b.movie, "'", ''), '"|\\.|\\)|\\`', '')), LOWER(a.text)) AS pred_overlaps
FROM max_token_counts a
LEFT JOIN user_nsulliv3.simplER_imdb b
ON a.id = b.id
WHERE a.id NOT IN (SELECT * FROM ids_with_match)
AND a.max_occur = a.n_occur)

SELECT COUNT(DISTINCT id)
FROM entities_overlap
WHERE pred_overlaps = true;
''')
