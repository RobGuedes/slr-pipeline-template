"""Step 4 — Preprocessing: NLP pipeline for topic modeling.

Handles text cleaning, tokenization, lemmatization (NLTK), and
corpus creation (Gensim).
"""

from __future__ import annotations

from collections import OrderedDict

import nltk
from gensim.corpora import Dictionary
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


def setup_nltk() -> None:
    """Download required NLTK data (idempotent)."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)


# Module-level tokenizer instance (stateless, safe to reuse)
_tokenizer = RegexpTokenizer(r"\w+")


def _get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tag to WordNet POS for lemmatization."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text: str) -> list[str]:
    """Clean, tokenize, POS-lemmatize, and deduplicate a single document.

    Pipeline (matches legacy notebook methodology):
    1. Lowercase & unidecode (remove accents)
    2. Tokenize with RegexpTokenizer(r'\\w+')
    3. Remove stopwords
    4. POS-tag and lemmatize with correct part-of-speech
    5. Remove duplicate tokens (preserve first-occurrence order)
    6. Remove short tokens (len <= 2)

    Parameters
    ----------
    text : str
        Raw text (e.g. abstract + title).

    Returns
    -------
    list[str]
        List of cleaned, unique tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. Normalize
    text = unidecode(text).lower()

    # 2. Tokenize (keeps digits-in-words like lstm2, t5)
    tokens = _tokenizer.tokenize(text)

    # 3. Stopwords
    stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stops]

    # 4. POS-aware lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(token, _get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]

    # 5. Deduplicate (preserve order)
    tokens = list(OrderedDict.fromkeys(tokens))

    # 6. Length filter
    tokens = [t for t in tokens if len(t) > 2]

    return tokens


def create_corpus(
    docs: list[list[str]],
    no_below: int = 5,
    no_above: float = 0.5,
) -> tuple[Dictionary, list[list[tuple[int, int]]]]:
    """Create a Gensim Dictionary and Bow Corpus from tokenized docs.

    Parameters
    ----------
    docs : list[list[str]]
        List of tokenized documents.
    no_below : int
        Keep tokens which are contained in at least `no_below` documents.
    no_above : float
        Keep tokens which are contained in no more than `no_above`
        (fraction) of the documents.

    Returns
    -------
    (Dictionary, Corpus)
        Gensim Dictionary and list of bag-of-words vectors.
    """
    if not docs:
        return Dictionary([]), []

    # Create dictionary
    dictionary = Dictionary(docs)

    # Filter extremes (legacy notebook did this)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # Convert to standard python list (bow) immediately?
    # Gensim doc2bow returns list of (id, count)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus
