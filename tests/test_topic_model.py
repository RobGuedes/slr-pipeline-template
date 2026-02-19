"""Tests for pipeline.topic_model — LDA hyperparameter sweep."""

import pytest
from unittest.mock import MagicMock, patch
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from pipeline.topic_model import perform_lda_sweep, select_top_candidates, train_final_model, SweepResult
from pipeline.config import PipelineConfig

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_dictionary():
    """Create a minimal Gensim dictionary."""
    docs = [["machine", "learning"], ["deep", "learning"], ["topic", "model"]]
    return Dictionary(docs)


@pytest.fixture
def mock_corpus(mock_dictionary):
    """Create a minimal corpus."""
    docs = [["machine", "learning"], ["deep", "learning"], ["topic", "model"]]
    return [mock_dictionary.doc2bow(doc) for doc in docs]


@pytest.fixture
def mock_texts():
    """Tokenized texts for c_v coherence."""
    return [["machine", "learning"], ["deep", "learning"], ["topic", "model"]]


# ── perform_lda_sweep ─────────────────────────────────────────────────


class TestPerformLdaSweep:
    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_sweeps_range_of_k(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify it iterates over the requested range of K."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.return_value = -7.0
        MockLda.return_value = mock_lda_instance

        mock_coherence_instance = MagicMock()
        mock_coherence_instance.get_coherence.return_value = 0.5
        MockCoherence.return_value = mock_coherence_instance

        k_values = [2, 3]
        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=k_values,
            passes=1,
            random_state=42,
        )

        assert len(results) == 2
        assert results[0].k == 2
        assert results[1].k == 3
        assert results[0].perplexity == -7.0
        assert results[1].perplexity == -7.0
        assert MockLda.call_count == 2

        call_args_list = MockLda.call_args_list
        assert call_args_list[0].kwargs["num_topics"] == 2
        assert call_args_list[1].kwargs["num_topics"] == 3

    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_uses_c_v_coherence(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify CoherenceModel is called with coherence='c_v' and texts."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.return_value = -7.0
        MockLda.return_value = mock_lda_instance

        mock_coherence_instance = MagicMock()
        mock_coherence_instance.get_coherence.return_value = 0.5
        MockCoherence.return_value = mock_coherence_instance

        perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=[2],
            passes=1,
        )

        # Verify CoherenceModel was called with c_v and texts
        MockCoherence.assert_called_once()
        call_kwargs = MockCoherence.call_args.kwargs
        assert call_kwargs["coherence"] == "c_v"
        assert call_kwargs["texts"] == mock_texts

    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_returns_coherence_and_perplexity(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify it captures coherence score and perplexity."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.side_effect = [-7.0, -6.5]
        MockLda.return_value = mock_lda_instance

        mock_coherence = MockCoherence.return_value
        mock_coherence.get_coherence.side_effect = [0.4, 0.6]

        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=[2, 3],
            passes=1,
        )

        assert results[0].coherence == 0.4
        assert results[1].coherence == 0.6
        assert results[0].perplexity == -7.0
        assert results[1].perplexity == -6.5


# ── SweepResult ───────────────────────────────────────────────────────


class TestSweepResult:
    def test_sweep_result_stores_metrics_only(self):
        """SweepResult should NOT store the full LdaModel object."""
        result = SweepResult(k=5, coherence=0.42, perplexity=-7.1)
        assert result.k == 5
        assert result.coherence == 0.42
        assert result.perplexity == -7.1
        assert not hasattr(result, "model")


# ── select_top_candidates ─────────────────────────────────────────────


class TestSelectTopCandidates:
    def test_returns_top_n_by_coherence(self):
        """Top 3 from 5 results, sorted by coherence descending."""
        results = [
            SweepResult(k=2, coherence=0.30, perplexity=-7.0),
            SweepResult(k=3, coherence=0.50, perplexity=-6.5),
            SweepResult(k=4, coherence=0.45, perplexity=-6.8),
            SweepResult(k=5, coherence=0.60, perplexity=-6.2),
            SweepResult(k=6, coherence=0.40, perplexity=-6.9),
        ]
        top = select_top_candidates(results, n=3)
        assert len(top) == 3
        assert top[0].k == 5   # highest coherence
        assert top[1].k == 3   # second
        assert top[2].k == 4   # third

    def test_returns_all_if_fewer_than_n(self):
        """If only 2 results exist, return both (don't crash)."""
        results = [
            SweepResult(k=2, coherence=0.30, perplexity=-7.0),
            SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        ]
        top = select_top_candidates(results, n=3)
        assert len(top) == 2
        assert top[0].k == 3
        assert top[1].k == 2

    def test_default_n_is_3(self):
        """Verify default n=3 without passing argument."""
        results = [
            SweepResult(k=i, coherence=i * 0.1, perplexity=-7.0)
            for i in range(2, 8)
        ]
        top = select_top_candidates(results)
        assert len(top) == 3


# ── train_final_model ─────────────────────────────────────────────────


class TestTrainFinalModel:
    def test_returns_lda_model(self, mock_corpus, mock_dictionary):
        """Integration style test with tiny data/iterations."""
        model = train_final_model(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            num_topics=2,
            passes=1,
            random_state=42
        )
        assert isinstance(model, LdaModel)
        assert model.num_topics == 2
