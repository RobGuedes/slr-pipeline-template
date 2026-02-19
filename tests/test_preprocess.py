"""Tests for pipeline.preprocess — text cleaning and Gensim dictionary/corpus creation."""

import pytest
from gensim.corpora import Dictionary

from pipeline.preprocess import clean_text, create_corpus, setup_nltk


# ── setup_nltk ─────────────────────────────────────────────────────────


def test_setup_nltk_downloads_pos_tagger():
    """setup_nltk should make POS tagger available."""
    setup_nltk()
    import nltk
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")


# ── clean_text ────────────────────────────────────────────────────────


class TestCleanText:
    @pytest.fixture(scope="class", autouse=True)
    def setup_nltk_data(self):
        """Ensure NLTK data is present for tests."""
        setup_nltk()

    def test_lowercases(self):
        assert clean_text("Hello World") == ["hello", "world"]

    def test_removes_punctuation(self):
        assert clean_text("hello, world!") == ["hello", "world"]

    def test_removes_numbers(self):
        assert clean_text("hello 123 world") == ["hello", "world"]

    def test_lemmatizes(self):
        # Default WordNetLemmatizer treats words as nouns.
        # "cats" -> "cat"
        res = clean_text("cats")
        assert "cat" in res

    def test_removes_stopwords(self):
        # "the" is a standard stopword
        assert clean_text("the cat") == ["cat"]

    def test_removes_short_tokens(self):
        # "is", "a" are sort, "go" is 2 chars.
        assert clean_text("ab cde") == ["cde"]

    def test_handles_empty(self):
        assert clean_text("") == []

    def test_return_type_is_list(self):
        assert isinstance(clean_text("test"), list)


# ── create_corpus ─────────────────────────────────────────────────────


class TestCreateCorpus:
    def test_returns_dictionary_and_corpus(self):
        docs = [["machine", "learning"], ["deep", "learning"]]
        dictionary, corpus = create_corpus(docs)
        assert isinstance(dictionary, Dictionary)
        assert len(corpus) == 2

    def test_filter_extremes(self):
        # 'common' appears in 2 docs. 'rare' in 1.
        # If no_below=2, 'rare' should be filtered out.
        docs = [["common", "word"], ["common", "thing"], ["rare"]]
        # no_above=1.0 needed because common is in 66% of docs
        dictionary, corpus = create_corpus(docs, no_below=2, no_above=1.0)
        assert "common" in dictionary.token2id
        assert "rare" not in dictionary.token2id

    def test_filter_most_common(self):
        # 'common' appears in 100% of docs. If no_above=0.5, strictly it should go?
        # Gensim no_above is fraction of total corpus size.
        docs = [["common", "a"], ["common", "b"], ["common", "c"]]
        dictionary, corpus = create_corpus(docs, no_above=0.5)
        assert "common" not in dictionary.token2id

    def test_dictionary_is_compactified(self):
        docs = [["a", "b"], ["b", "c"]]
        dictionary, _ = create_corpus(docs)
        # IDs should be contiguous 0..N
        assert sorted(dictionary.keys()) == list(range(len(dictionary)))
