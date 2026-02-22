## **3\. Methodology**

This study follows the structure of a standalone systematic literature review, as classified by Xiao & Watson (2017), relying primarily on computational text analysis. The review was conducted in nine distinct steps, adapted from the methodologies proposed by Xiao & Watson (2017) and Asmussen & Møller (2019), grouped into three phases: planning, execution, and reporting (Figure 1). Steps 1 and 2 are performed manually by the researcher; steps 3 through 9 are executed using a Python pipeline developed for this purpose.

### **3.1. Formulation**

The research problem and scope are defined by the researcher, including the central research question, relevant subquestions, and the conceptual boundaries of the review domain. This step produces the rationale for all subsequent methodological choices.

### **3.2. Protocol Development**

A review protocol is designed to ensure unbiased and reproducible execution across all stages. The protocol defines inclusion and exclusion criteria, search databases, keyword combinations, and thresholds for data filtering. It also specifies the time range of publications to be retrieved from each database. All data processing and analysis are performed using Python 3.13.

### **3.3. Literature Search**

Data collection is conducted using Scopus and Web of Science (WoS). Searches target peer-reviewed journal articles in English, excluding conference papers, book chapters, and other non-journal sources, unless otherwise specified in the protocol. Keywords are combined using Boolean operators to form search queries, which are executed in each database independently. The resulting exports — CSV files from Scopus and tab-delimited TXT files from WoS — are loaded, merged, and deduplicated by the pipeline. Deduplication applies two complementary strategies: exact DOI matching (for records with a valid DOI) and normalized title combined with first-author surname matching (for records without a DOI or with conflicting identifiers). When duplicates are found, the citation count is averaged across the duplicate records. The temporal scope of the search is defined in the protocol (step 3.2) and applied at the database query stage, not by the pipeline.

### **3.4. Preprocessing**

The preprocessing phase prepares document text for topic modeling. The pipeline concatenates each document's title and abstract into a single text field. Preprocessing is performed using the NLTK library (Bird et al.) and includes: Unicode normalization, lowercasing, tokenization using a word-boundary regular expression, English stopword removal, part-of-speech tagging, and WordNet lemmatization with POS-aware morphological reduction. Duplicate tokens within a document are removed to reduce repetition bias, and tokens shorter than three characters are discarded. The resulting token sequences are used to construct a Gensim dictionary and bag-of-words corpus. Vocabulary is further filtered by removing tokens that appear in fewer than five documents or in more than 50% of the corpus.

### **3.5. Topic Modeling**

Latent Dirichlet Allocation (LDA), as described by Blei et al. (2003), is employed for topic modeling using the Gensim library (Řehůřek & Sojka, 2010). Coherence (c\_v) and log-perplexity scores are computed for models trained across a range of topic counts K (default: 2 to 15). All sweep models are trained with a fixed random seed to ensure reproducibility. Coherence and perplexity curves are saved as a diagnostic chart, and the top candidate values of K are presented to the researcher, who selects the optimal K prior to final model training.

### **3.6. Topic Identification**

The final LDA model is trained with an increased number of passes (default: 40\) over the corpus to improve convergence. The top 10 most important terms and their associated weights are extracted for each topic. The researcher is responsible for inspecting these terms — together with the interactive topic map generated in step 3.9 — and assigning a descriptive label to each topic. An LLM (e.g., OpenAI ChatGPT 5.2, Claude Sonnet. 4.6, Gemini 3 Pro) may be used as an aid to propose label candidates based on the extracted terms; however, final label assignment is a human decision.

### **3.7. Topic Selection**

Topics identified in step 3.6 are evaluated by the researcher for relevance to the research question, using the pyLDAvis interactive topic map as the primary visual tool. For each retained topic, the pipeline assigns a dominant topic and a membership probability to every document. Document filtering is then performed using a two-pass process. First, documents are retained if their probability of belonging to the dominant topic exceeds a threshold (default: 70%) and if they have accumulated at least a configurable minimum number of citations (default: 10). Per-topic citation thresholds may be set independently to account for variation in topic maturity or citation norms. Second, a recency-aware citation recovery pass is applied to recover temporally recent research that may not have accumulated sufficient citations yet. Very recent papers (e.g., < 2 years) can be recovered even with 0 citations if they are published by top-15 authors or in top-15 publication sources, while mid-range recent papers (e.g., 2-6 years) can be recovered if they meet a lowered, secondary citation threshold.

### **3.8. Quality Assessment**

As an additional quality control measure, the pipeline exports the filtered document set to a CSV file with a `Keep` column for manual annotation. The researcher reviews the abstract of each selected document and marks it for retention or rejection. The pipeline then loads the annotated file and carries forward only the documents confirmed by the researcher. This step validates the quantitative text-mining selections made in earlier stages.

### **3.9. Synthesis**

Global bibliometric characteristics of the confirmed document set are analyzed, drawing inspiration from methods in the LitStudy library (Heldens et al., 2022) but implemented via custom matplotlib visualizations sharing a unified visual identity. These include trends in annual publications, authors with the highest number of publications, top publication sources, and countries and institutions with significant representation. An interactive multidimensional topic map is generated using the pyLDAvis library (Sievert & Shirley, 2014), enabling visual exploration of topic relationships and salient term associations. Per-topic synthesis — identifying domain-specific findings, methods, and performance outcomes within each topic cluster — is performed manually by the researcher using the confirmed documents as input.
