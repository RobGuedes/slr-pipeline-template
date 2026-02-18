"""Step 4 — Preprocessing: NLP pipeline for topic modeling.

Handles text cleaning, tokenization, lemmatization (NLTK), and
corpus creation (Gensim).
"""

from __future__ import annotations

import re
import warnings

import nltk
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
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


def clean_text(text: str) -> list[str]:
    """Clean, tokenize, and lemmatize a single document string.

    Pipeline:
    1. Lowercase & unidecode (remove accents)
    2. Keep only alphanumeric chars (remove punctuation/numbers)
    3. Remove stopwords
    4. Lemmatize (nouns, verbs)
    5. Remove short tokens (< 3 chars)

    Parameters
    ----------
    text : str
        Raw text (e.g. abstract + title).

    Returns
    -------
    list[str]
        List of cleaned tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. Normalize
    text = unidecode(text).lower()

    # 2. Tokenize (keep only words)
    # Legacy used: re.sub('[^a-zA-Z]', ' ', text) then split
    # We'll use a regex tokenizer for better control
    tokens = re.findall(r"\b[a-z]{2,}\b", text)

    # 3. Stopwords
    stops = set(stopwords.words("english"))
    # Add custom stopwords from legacy if needed (none explicit in scan)
    tokens = [t for t in tokens if t not in stops]

    # 4. Lemmatize
    lemmatizer = WordNetLemmatizer()
    # NLTK lemmatize defaults to noun (n). Legacy likely used default.
    # We'll do noun then verb for better coverage, or just noun?
    # Legacy notebook: `lemma = WordNetLemmatizer(); lemma.lemmatize(word)` -> noun default
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 5. Length check (legacy implicit in some regexes, we explicit > 2)
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
