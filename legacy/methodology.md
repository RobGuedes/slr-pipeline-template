## **3\. Methodology**

This study falls under the category of a testing stand-alone review, as classified by Xiao & Watson (2017), and is characterized by the predominant use of statistical analyses. To achieve the research objective, the literature review was conducted in nine distinct steps, adapted from the methodologies proposed by Xiao & Watson (2017) and Asmussen & Møller (2019). These steps are illustrated in Figure 1 and grouped into three main phases: planning, execution, and reporting.  

### *3.1. Formulation*

As discussed in Chapter 2, portfolio optimization represents a prominent research topic in both academic literature and the financial industry, with growing adoption and development of techniques to enhance performance. The research problem was framed as follows: “What optimization methods can lead to allocative efficiency in investment portfolios?” To address this question, a systematic literature review was conducted with the aim of identifying techniques that have demonstrated superior performance in portfolio management.

### *3.2. Protocol development*

The protocol for this systematic literature review was designed to ensure unbiased execution and methodological rigor across all stages. Inclusion and exclusion criteria, search parameters, and procedures for data collection, analysis, and synthesis were established. Inspired by the methodologies of Xiao & Watson (2017) and Asmussen & Møller (2019), the protocol sought to achieve both comprehensive coverage and data manageability by defining databases, keywords, and evaluation criteria. All data processing and analysis were performed using Python version 3.12.

### *3.3. Literature search*

Data collection was conducted using Scopus and Web of Science, targeting primarily peer-reviewed journal articles in English, while excluding conference papers, book chapters, and other non-journal sources. Keywords and phrases were combined using Boolean operators to create multiple search queries, which were executed concurrently to facilitate preprocessing. Duplicate records from both databases were identified and removed using the ScientoPy library (RuizRosero et al., 2019). To limit the results to a manageable number, the search was refined iteratively, including filtering publications from 2019 onwards.

### *3.4. Preprocessing*

The preprocessing phase utilized the NLTK library for tasks such as partof-speech tagging, tokenization, lemmatization, and importing stopwords (Bird et al.). TF-IDF vectorization was performed using the Scikit-learn library, ensuring that the text data were adequately prepared for analysis (Pedregosa et al., 2011).

### *3.5. Topic modeling*

Latent Dirichlet Allocation (LDA\[GA3\] ), as described by Blei et al. (2003) and Guindani et al. (2024), was employed for topic modeling. Abstracts were preprocessed and analyzed using the Gensim library, with coherence and perplexity scores calculated for topic numbers ranging from 2 to 15 Reh˚uˇrek & Sojka (2010). To enhance reproducibilityˇ and minimize variability, all LDA models were executed with 20 passes over the corpus and a fixed random seed. The optimal number of topics was determined based on coherence and perplexity scores. Topic visualizations, including multidimensional maps and salient term associations, were generated using the pyLDAvis library (Sievert & Shirley, 2014).

### *3.6. Topic identification*

To ensure accuracy, the number of passes over the corpus was increased to 40 for final LDA runs. The top 10 most important terms for each topic, along with their respective weights, were extracted. The o1-preview generative AI model from OpenAI was employed to propose topic labels based on these terms (OpenAI, 2024).

### *3.7. Topic selection*

Topics generated in the previous step were evaluated by the authors to determine their relevance to the research objectives. Documents were retained within each topic if their probability of belonging to the topic exceeded 70% and if they had at least 15 citations. This filtering process ensured the inclusion of highly relevant and impactful documents.

### *3.8. Quality assessment*

As an additional quality control measure, the abstracts of all selected documents were individually reviewed to validate the quantitative text-mining selections made in earlier stages.

### *3.9. Synthesis*

Global characteristics of the dataset, such as trends in the number of annual publications, authors with the highest number of publications, and countries and institutions with significant representation, were analyzed using the ScientoPy (Ruiz-Rosero et al., 2019\) and LitStudy libraries (Heldens et al., 2022). Furthermore, the selected articles within each topic were examined to identify portfolio optimization methods and their respective performance outcomes.