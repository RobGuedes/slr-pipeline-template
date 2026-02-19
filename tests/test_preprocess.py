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
        tokens = clean_text("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_punctuation(self):
        # RegexpTokenizer(r'\w+') splits on non-word chars
        tokens = clean_text("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_keeps_digits_in_words(self):
        # RegexpTokenizer keeps 'lstm2', 't5' etc.
        tokens = clean_text("lstm2 model works")
        assert "lstm2" in tokens

    def test_lemmatizes_verbs_with_pos(self):
        # POS-aware: "running" (verb) -> "run"
        tokens = clean_text("The cats are running quickly")
        assert "cat" in tokens
        assert "run" in tokens

    def test_lemmatizes_nouns(self):
        # "studies" -> "study" (noun), "cats" -> "cat" (noun)
        tokens = clean_text("multiple studies about cats")
        assert "study" in tokens
        assert "cat" in tokens

    def test_removes_stopwords(self):
        tokens = clean_text("the cat sat on the mat")
        assert "the" not in tokens
        assert "cat" in tokens

    def test_removes_short_tokens(self):
        # Tokens with len <= 2 are removed after lemmatization
        tokens = clean_text("ab cde fg")
        assert "ab" not in tokens
        assert "fg" not in tokens
        assert "cde" in tokens

    def test_deduplicates_tokens_per_document(self):
        # Same word repeated should appear only once
        tokens = clean_text("model model model training model")
        assert tokens.count("model") == 1

    def test_dedup_preserves_order(self):
        # First occurrence order should be preserved
        tokens = clean_text("alpha beta alpha gamma beta")
        assert tokens.index("alpha") < tokens.index("beta")
        assert tokens.index("beta") < tokens.index("gamma")

    def test_handles_empty(self):
        assert clean_text("") == []

    def test_handles_none_like(self):
        assert clean_text("   ") == []

    def test_return_type_is_list(self):
        assert isinstance(clean_text("test something here"), list)


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
