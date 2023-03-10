# Databricks notebook source
!pip install flair
import pandas as pd
from flair.nn import Classifier
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter
from utils.simplER import *

# -------------
# Load model
# -------------

tagger = Classifier.load('ner-ontonotes-large')

# ---------
# Data
# ---------

imdb_reviews = pd.read_csv('https://raw.githubusercontent.com/sullivannicole/simplER/main/data/imdb_reviews.csv')
imdb_reviews = imdb_reviews.rename(columns={imdb_reviews.columns[0]: 'id'})

# spark.createDataFrame(imdb_reviews).write.mode('overwrite').saveAsTable('user_nsulliv3.simplER_imdb')

rvw_text = imdb_reviews.user_review.values
split_df = split_sentences(rvw_text)
ner_results = run_NER(split_df)

# spark.createDataFrame(ner_results).write.mode('overwrite').saveAsTable('user_nsulliv3.simplER_imdb_NER')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **Table creation:**
# MAGIC 
# MAGIC 1. simpler_imdb_no_punc: remove all alphanumerics out of ground truth lable
# MAGIC 2. simpler_imdb_eval: contains most of the matches via exact & overlap methods
# MAGIC 3. simpler_imdb_manual_ided_matches: manually ID'd some add'tl matches (not necessarily majority, but in most cases they are)
# MAGIC 4. simpler_imdb_eval_maj_ranks: ranks for *majority votes* only

# COMMAND ----------

# ---------------
# Evaluation
# ---------------

# 1. Strip all characters other than alphanumerics out of ground truth label + save as extra column
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_no_punc
									SELECT *, LOWER(RTRIM(REGEXP_REPLACE(movie, '[^0-9a-zA-Z ]', ''))) AS movie_no_punc
									FROM user_nsulliv3.simpler_imdb;
''')

# 2a. Create eval table w/ ground truth and predictions
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_eval AS
									WITH art_ents AS (SELECT id, LOWER(RTRIM(REGEXP_REPLACE(text, '[^0-9a-zA-Z ]', ''))) AS text, score --,  RANK(score) OVER (PARTITION BY id ORDER BY score DESC) AS score_rank, score
									FROM user_nsulliv3.simplER_imdb_NER
									WHERE entity_detected = 'WORK_OF_ART'),
									
									token_counts AS (SELECT id, text, AVG(score) AS avg_score, COUNT(*) AS n_occur
									FROM art_ents
									GROUP BY 1, 2),
									
									max_token_counts AS (SELECT *, MAX(n_occur) OVER (PARTITION BY id) AS max_occur
									FROM token_counts)
									
									-- This excludes any IDs with NO work-of-art entities detected
									SELECT a.*, b.movie, b.movie_no_punc,
									
									CASE WHEN a.n_occur = a.max_occur AND a.text = b.movie_no_punc THEN 'exact'
									WHEN a.n_occur = a.max_occur AND (CONTAINS(b.movie_no_punc, a.text) OR CONTAINS(a.text, b.movie_no_punc)) THEN 'overlap'
									ELSE 'none' END AS majority_vote_match_type,
									
									CASE WHEN a.n_occur = a.max_occur AND a.text = b.movie_no_punc THEN 2
									WHEN a.n_occur = a.max_occur AND (CONTAINS(b.movie_no_punc, a.text) OR CONTAINS(a.text, b.movie_no_punc)) THEN 1 
									ELSE 0 END AS majority_vote_match_type_code,
									
									CASE WHEN a.n_occur = a.max_occur AND (a.text = b.movie_no_punc OR CONTAINS(b.movie_no_punc, a.text) OR CONTAINS(a.text, b.movie_no_punc)) THEN 1
									ELSE 0 END AS majority_vote_match,
									
									CASE WHEN a.text = b.movie_no_punc OR CONTAINS(b.movie_no_punc, a.text) OR CONTAINS(a.text, b.movie_no_punc) THEN 1 ELSE 0 END AS non_majority_vote_match
									
									FROM max_token_counts a
									LEFT JOIN user_nsulliv3.simplER_imdb_no_punc b
									ON a.id = b.id;
''')

# 2b. For purpose of time, manually ID'ed some matches not encompassed by above logic
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_manual_ided_matches
									SELECT *, n_occur AS rank_first_match
									FROM user_nsulliv3.simpler_imdb_eval
									WHERE id IN (
									89, 107, 126, 157, 160, 261, 264, 285, 297, 323, 339, 363, 365, 379, 443, 540, 549, 570,
									571, 586, 590, 593, 601, 604, 630, 631, 650, 675, 690, 698, 712, 761, 780, 794, 918, 958, 971, 1042, 1073,
									1155, 1222, 1233, 1239, 1262, 1279, 1291, 1336, 1383, 1389, 1451, 1465, 1475, 1480, 1498, 1584, 1598, 1608,
									1617, 1661, 1739, 1774, 1816, 1854, 1872, 1893, 1938, 1956
									)
									AND text IN (
									't2', 'capharnaum', 'dkr', 'drstrangelove', 'kgf chapter 2', 'the three billboard', 'ls2sb',
									'pan singh tomar', 'cdi', 'time of gypsies', 'yojinbo', 'judgment at nuremburg', 'rashmon', 'znmd', 'nausicaa of the valley of the wind', 'fanny och alexander',
									'cries and whispers', 'whatever happened to baby jane', 'tmwslv', 'childhood of ivan', 'grand illusion', 'bajrangibhaijaan', 'the raid redemption', 'tropa de elite 2', 'about elly',
									'tropa de elite', 'knockin on the heavens door', 'bibo', 'la rgle du jeu', 'x men', 'un prophte', 'trois couleurs blanc', 'all the presidens men', 'all the presidens men',
									'udtapunjab', 'the girl with a dragon tattoo', 'the boy in the striped pyjamas', 'auf der anderen seite', 'pride and prejudice', 'lilja 4ever', 'cthd', 'the double life of veronique', 
									'the outlaw josie wales', 'la planete sauvage', 'mi5', 'me earl', 'captain america winter soldier', 'wreck it ralph', 'withnail and i',
									'the lady killers', 'gotg 2', 'kick ass', 'hp', 'un long dimanche de fianailles', 'y tu mama tambien', 'dc', 'planes trains and automobiles');

''')

# -----------------------------
# Rankings for majority votes
# -----------------------------

# 1. Select majority votes only, with > 1 vote
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_eval_maj_ranks

                  -- 1. Take majority votes
                  WITH mrr1 AS (SELECT id, text, majority_vote_match, RANK(avg_score) OVER (PARTITION BY id ORDER BY n_occur DESC, avg_score DESC) AS relevance_rank
                                    FROM user_nsulliv3.simpler_imdb_eval
                                    WHERE n_occur = max_occur -- get majority votes only
                                    AND max_occur > 1
                                    ),
                                    
                  -- 2. Pull min rank of a match amongst majority votes (no manually IDed matches) --> rank of first correct answer
                  majority_ranks AS (SELECT id, MIN(relevance_rank) AS rank_first_match
                                    FROM mrr1
                                    WHERE majority_vote_match = 1
                                    GROUP BY 1)

                  -- 3. Get reciprocal rank					
                  SELECT DISTINCT id, rank_first_match, 1/rank_first_match AS reciprocal_rank 
                  FROM majority_ranks

''')

# 2a. MRR
spark.sql('''
SELECT AVG(reciprocal_rank) AS mrr
FROM user_nsulliv3.simpler_imdb_eval_maj_ranks;
''')

# 2c. Has pos@k

k = 1

spark.sql(f'''
-- Select movies with a WOA entity detected, must have more than 1 vote
WITH movies_w_woa AS (
SELECT COUNT(DISTINCT id) AS n_w_maj_votes
FROM user_nsulliv3.simpler_imdb_eval
WHERE n_occur = max_occur -- get majority votes only
AND id NOT IN (SELECT DISTINCT id FROM user_nsulliv3.simpler_imdb_manual_ided_matches)
AND max_occur > 1
),

-- Select majority votes of WOA, must have more than one vote
pos_at_k AS (SELECT COUNT(DISTINCT a.id) n_w_maj_and_match
FROM user_nsulliv3.simpler_imdb_eval_maj_ranks a
WHERE a.rank_first_match <= {k})

SELECT a.n_w_maj_and_match/b.n_w_maj_votes AS has_pos_at_k
FROM pos_at_k a
JOIN movies_w_woa b;
''')




# COMMAND ----------

# ------------------------------------------------------------------------
# Rankings incl. those with no match whatsoever (no WOA entities ID'ed)
# ------------------------------------------------------------------------

# 1. Create rankings
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_eval_ranks
									WITH mrr1 AS (SELECT id, text, non_majority_vote_match, RANK(avg_score) OVER (PARTITION BY id ORDER BY n_occur DESC, avg_score DESC) AS relevance_rank
									FROM user_nsulliv3.simpler_imdb_eval),
									
									match_ranks (SELECT id, MIN(relevance_rank) AS rank_first_match --MIN(relevance_rank) FILTER(WHERE (majority_vote_match) = (1)) AS rank_first_match
									FROM mrr1
									WHERE non_majority_vote_match = 1
									GROUP BY 1
									
									UNION ALL
									SELECT DISTINCT id, rank_first_match
									FROM user_nsulliv3.simpler_imdb_manual_ided_matches), -- union to manual matches here
									
									all_ranks AS (SELECT DISTINCT a.id, b.rank_first_match
									FROM user_nsulliv3.simpler_imdb a
									LEFT JOIN match_ranks b 
									ON a.id = b.id)
									
									SELECT DISTINCT id, CASE WHEN rank_first_match IS NULL THEN 0 ELSE rank_first_match END AS rank_first_match, 
									CASE WHEN rank_first_match IS NULL THEN 0 ELSE 1/rank_first_match END AS reciprocal_rank 
									FROM all_ranks;
;

''')

# 2a. MRR
spark.sql('''
SELECT AVG(reciprocal_rank) AS mrr
FROM user_nsulliv3.simpler_imdb_eval_ranks;
''')

# 2b. MAP@k

# 2c. Has positive @k
spark.sql('''
SELECT COUNT(DISTINCT id)
FROM user_nsulliv3.simpler_imdb_eval_ranks
WHERE rank_first_match <= 1;
''')

spark.sql('''
SELECT COUNT(DISTINCT id)
FROM user_nsulliv3.simpler_imdb_eval_ranks
WHERE rank_first_match <= 5;
''')

spark.sql('''
SELECT COUNT(DISTINCT id)
FROM user_nsulliv3.simpler_imdb_eval_ranks
WHERE rank_first_match <= 20;
''')
