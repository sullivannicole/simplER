# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Load packages + Dolly
# MAGIC 
# MAGIC Load [Databricks' Dolly](https://github.com/databrickslabs/dolly) from HuggingFace. This code works with Databricks LTS 11.3.

# COMMAND ----------

# -----------------
# GPU set-up
# -----------------

cuda = True # change to False if you only want to run on cpu
import torch
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# -----------------
# Imports
# -----------------

!pip install accelerate
import accelerate
import numpy as np
import pandas as pd
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer)

# -----------------
# Load Dolly
# -----------------

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side = "left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map = "auto", trust_remote_code = True, low_cpu_mem_usage = True)

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def generate_response(instruction: str, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda") # requires GPU

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

# Run this line if you get a maxed-out memory error:
# torch.cuda.empty_cache()

# COMMAND ----------

# Play around w/ Dolly to do the NER w/ a couple titles

user_prompt = '''If the the official title of the movie being reviewed exists in the following text, extract it and respond ONLY with the official movie title; otherwise, try to infer the movie title
'''

review = '''"Elia Kazan's film is still amazing after 50 years. It's curious how it parallels Kazan's own life in the way the main character, Terry Malloy, ends up naming names to the commission investigating the corruption on the waterfront, the same way the director did in front of the HUAC committee, presided by the evil Senator Joe McCarthy and his henchman, Roy Cohn. Bud Schulberg's screen play is his best work for the movies. It also helped that Elia Kazan had a free reign over the film, which otherwise could have gone wrong under someone else's direction.Terry Malloy, as we see him first, is a man without a conscience. He is instrumental in ratting on a fellow longshoreman, who is killed because he knows about the criminal activities on the piers. At the same time, Terry is transformed and ultimately redeems himself because he falls in love with Edie Doyle, the sister of the man that is killed by the mob. Terry Malloy is a complex character. His own brother Charley, is the right hand man of Johnny Friendly, the union boss. Charley is trying to save Terry. It's clear that Charley is going to be sacrificed because of the way he is acting by the same people he works with."'''

full_prompt = user_prompt + review

generate_response(full_prompt, model = model, tokenizer = tokenizer)

# COMMAND ----------

pd.read_csv(https://raw.githubusercontent.com/sullivannicole/simplER/main/data/imdb_after_gpt.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 1. NoTitle scenario
# MAGIC 
# MAGIC For the NoTitle scenario, we need to generate movie titles from table data using Dolly.

# COMMAND ----------

# Pull in TDmatch data
movie_tbl = pd.read_csv('https://raw.githubusercontent.com/naserahmadi/TDmatch/main/data/imdb/imdb_movielens.csv') 
spark.createDataFrame(movie_tbl).write.mode('overwrite').saveAsTable('user_nsulliv3.simpler_imdb_movies') # store locally for ease of use

spark.sql('''
create or replace table user_nsulliv3.simpler_imdb_movies as

select * except(year), year as release_yr
from user_nsulliv3.simpler_imdb_movies;
''')

# Create ground-truth table w/ IDs for data table since data table has 50k movies in it
spark.sql('''
CREATE OR REPLACE TABLE user_nsulliv3.simpler_imdb_groundtruth AS
select distinct a.id, a.movie, a.movie_no_punc, replace(replace(replace(a.movie_no_punc,' ','<>'),'><',''),'<>','_') AS movie_title_cw, b.title
from user_nsulliv3.simpler_imdb_no_punc a
left join user_nsulliv3.simpler_imdb_movies b 
on replace(replace(replace(a.movie_no_punc,' ','<>'),'><',''),'<>','_') = b.title;''')

# Create question for Dolly from table
dolly_qs = spark.sql('''
select a.title, concat_ws('', 'What ', coalesce(cast(a.release_yr as int), ''), ' movie', CONCAT(CONCAT(' was directed by ', INITCAP(REPLACE(a.director, '_', ' '))), ''), coalesce(concat(' and had actor(s) ', INITCAP(REPLACE(a.actor_1, '_', ' '))), ''), coalesce(concat(', ', INITCAP(REPLACE(a.actor_2, '_', ' '))), ''), coalesce(concat(', ', INITCAP(REPLACE(a.actor_3, '_', ' '))), ''), "? Respond with ONLY the movie's official title.") as dolly_q
from user_nsulliv3.simpler_imdb_movies a
inner join user_nsulliv3.simpler_imdb_groundtruth b 
on a.title = b.title
''')

dolly_qs_pd = dolly_qs.toPandas()

# COMMAND ----------

dolly_answers = []

for i in dolly_qs_pd['dolly_q'][:60]:
  dolly_answers.append(generate_response(i, model = model, tokenizer = tokenizer))

# dolly_qs_pd['dolly_prediction'] = dolly_answers

# spark.createDataFrame(dolly_qs_pd).write.mode('overwrite').saveAsTable('user_nsulliv3.simpler_imdb_dolly_preds')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Coronavirus text-data matching

# COMMAND ----------

covid_usr = spark.sql('''select *, CONCAT('Extract the country names (IGNORE continents) in this sentence; respond ONLY with the country names extracted: "', sentence, '"') AS dolly_q
from user_nsulliv3.simpler_covid_usr_corrected;''')

covid_usr_pd = covid_usr.toPandas()

# Predict
dolly_answers = []

for i in covid_usr_pd['dolly_q'][:10]:
  dolly_answers.append(generate_response(i, model = model, tokenizer = tokenizer))
