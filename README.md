# simplER

## Existing approaches + motivation
Much of the code underlying SOTA ER techniques are overly verbose, highly complicated and difficult to adapt to different domains. For example, TDMatch, a SOTA method for unsupervised matching of text to data, is an eyewatering 1,000+ lines of code - and incorporating external datasources necessitates using graph expansion and pruning techniques, both of which are computationally expensive. While TDmatch is a highly rigorous ER solution, it neglects to exploit existing SOTA tech available for free (spaCy, John Snow Labs, Google, ChatGPT, etc.), and also neglects the fact that many analysts and scientists in industry are expected to apply “appropriate level of effort” for problems - i.e., a solution that is very easy to implement and works for 90% of cases is almost always preferable to a solution that works for 99.9% of cases but is highly impractical to implement and difficult to maintain. Moreover, the rapid development of new technology and deprecation of what was once SOTA make a less-lengthy, more readable solution more desirable to the analyst - expending too much time and/or effort towards a solution, even if it’s highly robust, risk its being outdated before it’s ever even productionized.

Another shortcoming of existing approaches is that they assume neither of the datasets to be matched contain errors; however oftentimes data for these ER datasets is, at least initially, manually generated and could contain ER-affecting errors. Consider an Amazon dataset containing entries by some smaller sellers on the marketplace, who have manually typed out labels and descriptions for their products. One of the sellers offers different sizes of the same product, in 12.2 oz, and 8.6oz. However, in entering their descriptions, the seller misplaced the decimal and now it appears they offer a third product, of size 1.22oz. This is a critical error to detect for customers, as well as in order to merge the entities, and current SOTA ER methods are untested in this regard.

Finally, most ER methods require two datasets for matching to already exist; however, what if the analyst or scientist wants to generate the second dataset to which to match to **during the ER process?** This often occurs in cases where higher-level categorizations for products or items doesn’t already exist in a dataset but are desired for various use cases (marketing, analytics, sales, etc.). However, creating an exhaustive list of all the possible categories for a particular domain can be highly time-intensive and a data scientist or analyst may miss categories due to subjectivity. 

Consider the following example: a data scientist for a CPG company has a dataset, purchased, from an outside vendor, of products and their respective sales. A marketing partner would like to know the fastest growing “flavor spaces” for their food segment (cereal) in order to help proactively determine which of the company’s products to most aggressively market in the next quarter. However, the products dataset obtained from the vendor contains only product description and brand - it does not contain a “flavor” field. In this case, it would be useful to simultaneously generate the flavor dataset and match that dataset to the vendor’s dataset.

## Proposed solution
We propose a novel framework, **simplER**, which makes use of spaCy for named entity recognition (NER) and Google for document retrieval, exploiting the fact that many cases of non-sensitive entity resolution boil down to retrieving documents of high relevance to a string of named entities (notably, a great deal of sensitive data, such as full names, addresses, and ages of individuals is also freely available through these methods). These technologies are free and highly mature, making the entire pipeline easy for an analyst or scientist to implement and maintain for the foreseeable future. Moreover, the use of Google for document retrieval means that we can generate secondary datasets on the fly as needed, for free, with very little extra code. Currently our solution requires just 3 functions and an additional 10 lines of code to run the functions for a particular dataset. 

The contributions of our work are 2-fold: (1) our solution is highly compact (just 3 functions + 10 lines of code to run), doesn’t require any serious hardware (we ran on Colab with the free runtime), and uses free, mature SOTA tech, **making it easy to implement and maintain**; and (2) our solution is **able to generate secondary datasets on the fly to which the primary dataset can be matched.**

## Evaluation
We'll test our framework on 3 different scenarios, with and without an additional keyword in the Googling step, to demonstrate it's competitive or outperforms the baseline (TDmatch).

### Data to data
* Fodor-Zagats

### Unstructured text to data
* IMDB
* Snopes
* Politifact

### Data to generated data
* FDA food products dataset
