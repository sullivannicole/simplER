# simplER
 
## Existing approaches + shortcomings
Much of the code underlying state-of-the-art (SOTA) entity-resolution (ER) techniques are overly verbose, highly complicated and difficult to adapt to different domains. For example, TDMatch, a SOTA method for unsupervised matching of text to data, is an eyewatering 1,000+ lines of code - and incorporating external datasources necessitates using graph expansion and pruning techniques, both of which are computationally expensive. While TDmatch is a highly rigorous ER solution in the unsupervised context, it neglects to exploit existing SOTA technologies that have carefully been optimized for performance and runtime and are freely available (spaCy, John Snow Labs, Google, ChatGPT, etc.). It also neglects the fact that many analysts and scientists in industry are expected to apply “appropriate level of effort” for problems - i.e., a solution that is very easy to implement and works for 90% of cases is almost always preferable to a solution that works for 99.9% of cases but is highly impractical to implement and difficult to maintain. Finally, the rapid development of new technology quickly leads to these highly complicated, custom solutions being deprecated in favor of an even more robust competitor, making a less-lengthy, more readable solution highly desirable to the analyst - expending too much time and/or effort towards an expansive solution, despite its advantages, risk its being outdated before it’s ever even productionized.
 
A second drawback of using TDMatch is that the optional graph expansion that is built into the method only allows the user to reference general-knowledge databases, such as DBpedia. While this is useful for domains where entities are more well-known (celebrities, movie titles, cities/locations), for domains where the data are “sensitive” or wouldn’t appear in a public-knowledge database, utilizing such an external datasource won’t yield any gains in performance. However, especially in industry, many use cases for entity resolution involve entities that won’t appear in such databases.
 
Consider the following example:
> A local municipality has implemented a new short term rental regulation and would like to identify Airbnb owners in their city that they need to comply with the regulation within 90 days. They’ve purchased a set of Airbnb listings for their municipality from a vendor who scraped them from the website; however, the locations are inexact. They contain lat, lon coordinates referring to the center radius on the map displayed for their listing on the Airbnb website but it is unknown how far the actual property is from these coordinates. Moreover, the municipality has access to the assessor’s parcel data and owner names for all the properties within their boundaries. However, they have a limited budget to implement a framework to match these two entities, and a very limited budget for contracting workers to manually check the listing photos against Google Street View to ensure the property has been correctly identified. Therefore, they’d like to limit the number of listings they send to a human-in-the-loop to a very small subset. They also recognize that there are many cases where it’s difficult or impossible to resolve the two entities from photos alone (i.e. if there are no exterior photos on the Airbnb listing, or if the exterior is highly obscured), in which case the ER framework they implement is the only way to identify owner/address information.
 
Another shortcoming of existing approaches is that many assume neither of the datasets to be matched contain errors, or at the very least that the errors don’t impact the quality of the ER results; however oftentimes data for these ER datasets is, at least initially, manually-generated and could contain ER- affecting errors. ChatGPT is a new, free SOTA technology that will likely outperform current SOTA error- detection methods, given that it's trained on billions of highly-contextualized examples; therefore, with our ultimate aim of implementing a simplified ER framework that's easy for an analyst or scientist to implement, integrating ChatGPT error-detection is preferable to integrating other available error- detection techniques. To demonstrate the need for this step, consider, for example, an Amazon dataset containing entries by some smaller sellers on the marketplace, who have manually typed out labels and descriptions for their products. One of the sellers offers different sizes of the same product, in 12.2 oz, and 8.6oz. However, in entering their descriptions, the seller misplaced the decimal and now it appears they offer a third product, of size 1.22oz. This is a critical error to detect for customers, as well as in order to merge the entities, and current SOTA ER methods aren’t widely tested in the presence of errors.
 
Finally, most ER methods require two datasets for matching to already exist; however, what if the analyst or scientist wants to generate the second dataset to which to match to **during the ER process?** This can often occur when data enhancement is a desire, but can also be the case when entity identification must first be performed within a dataset. Take for example, a small movie theater chain in the Midwest that has access to a dataset of movie reviews that doesn’t contain any additional information about the movie being reviewed (title, genre, rating); it would be beneficial for the theater for marketing purposes to match these reviews to a dataset containing title, genre, rating etc. Rather than searching for a database of movies with this information (which could contain hundreds of thousands of entities and require careful blocking), we argue that for their use case the most efficient approach to simply resolve the entity of each review within the dataset and then surface the generate the secondary dataset on the fly.
 
Another context where generating the secondary dataset would benefit is in higher-level categorizations for products or items don’t already exist in a dataset but are desired for various use cases (marketing, analytics, sales, etc.). However, creating an exhaustive list of all the possible categories for a particular domain can be highly time-intensive and a data scientist or analyst may miss categories due to subjectivity. As far as we are aware, this use-case is yet to be addressed in existing ER literature, as it blends the lines between entity identification, entity matching and data enhancement.
 
Consider the following example: 
> A data scientist for a CPG company has a dataset, purchased from an outside vendor, of products and their respective sales. A marketing partner would like to know the fastest growing “flavor spaces” for their food segment (cereal) in order to help proactively determine which of the company’s products to most aggressively market in the next quarter. However, the products dataset obtained from the vendor contains only product description and brand - it does not contain a “flavor” field. In this case, it would be useful to simultaneously generate the flavor dataset and match that dataset to the vendor’s dataset.
 
Therefore our proposed solution has four desiderata: (1) easy implementation and maintenance, (2) ability to handle “sensitive” entities; (3) incorporation of SOTA error detection, and (4) ability to generate secondary datasets on the fly.
 
## Proposed solution
We propose a novel framework, simplER, which makes use of flair for named entity recognition (NER) and SOTA search engines for document retrieval, exploiting the fact that many cases of non-sensitive entity resolution boil down to retrieving documents of high relevance to a string of named entities (notably, a great deal of sensitive data, such as full names, addresses, and ages of individuals is also freely available through these methods). These technologies are free and highly mature, making the entire pipeline easy for an analyst or scientist to implement and maintain for the foreseeable future. Moreover, the use of a search engine for document retrieval has the added benefit of allowing the user to generate secondary datasets on the fly or to perform general data enhancement as needed, for free, with very little extra code. Note that shifting the task in this way allows us to “build in” blocking using the search engine's well-tuned relevance algorithm, and that it also makes it possible to incorporate “sensitive” information that isn’t available in public databases such as DBpedia. Lastly, we also integrate error-detection with ChatGPT into our pipeline; while a nascent technology, ChatGPT has been trained on billions of examples and has a great amount of yet-untested potential for error-detection, a much- needed addition to current ER frameworks.
 
## Novel contributions
The contributions of our work are 3-fold: (1) our solution will be highly compact, won’t require any serious hardware, and will use free, mature SOTA tech, making it easy to implement and maintain; (2) our solution will integrate error-detection and simultaneously benchmark ChatGPT's abilities against current existing error-detection methods; (3) our solution will be able to generate secondary datasets on the fly to which the primary dataset can be matched; and (4) our framework extends beyond public data to “sensitive” or highly local entities.

## Evaluation
We'll test our framework on 5 different scenarios, with and without error detection, to demonstrate it's competitive or outperforms the baseline (TDmatch).
 
### Data to data
* Fodor-Zagats

### Unstructured text to data
* IMDB

### Text to text
* Snopes
* Politifact

### “Sensitive” spatial data to “sensitive” spatial data (a novel context)
* Airbnb scraped listing - Assessor's parcel data

### Data to generated data (a novel context)
* FDA food products dataset

