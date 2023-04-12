## simplER
simplER is a novel easy-to-use framework that makes use of Flair for named entity recognition (NER) and GPT-3.5 for text-data and data-data matching contexts. By first passing observations through Flair, we ensure that only the most difficult-to-classify observations are passed onto GPT-3.5, reducing runtime and increasing the efficiency of our overall framework.  Notably, these technologies are free and user-friendly, making the entire pipeline easy for an analyst or scientist to implement and maintain for the foreseeable future.

## Novel contributions 
The contributions of our work are 3-fold: 
(1) our solution is highly compact (just ~15 lines of code at its essence) while still minimizing dependencies, doesn’t require any serious hardware (we ran on Colab with the free runtime), and uses free SOTA tech, making it **easy to implement and maintain**; 
(2) our framework includes a **novel evaluation of GPT-3.5** on the **hardest text-data** and **data-data matching** tasks (where existing NER solutions fail) and 
(3) our framework extends beyond public data to **"sensitive"/highly local entities** (not possible with TDmatch, the baseline method).

## Evaluation
We evaluated simplER in both the text-data matching context as well as the data-data matching context to demonstrate it's competitive or outperforms the baseline (TDmatch).

**Data to data**
* Fodor-Zagats

**"Sensitive" spatial data to "sensitive" spatial data**
* Airbnb-assessors

**Unstructured text to data**
* IMDB
* CoronaCheck

