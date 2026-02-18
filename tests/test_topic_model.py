"""Tests for pipeline.topic_model — LDA hyperparameter sweep."""

import pytest
from unittest.mock import MagicMock, patch
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from pipeline.topic_model import perform_lda_sweep, train_final_model
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


# ── perform_lda_sweep ─────────────────────────────────────────────────


class TestPerformLdaSweep:
    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_sweeps_range_of_k(self, MockCoherence, MockLda, mock_corpus, mock_dictionary):
        """Verify it iterates over the requested range of K."""
        # Setup mocks
        mock_lda_instance = MagicMock()
        MockLda.return_value = mock_lda_instance
        
        mock_coherence_instance = MagicMock()
        mock_coherence_instance.get_coherence.return_value = 0.5
        MockCoherence.return_value = mock_coherence_instance

        # Run sweep for K=2, 3
        k_values = [2, 3]
        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            k_values=k_values,
            passes=1,
            random_state=42
        )

        # Asserts
        assert len(results) == 2
        assert results[0].k == 2
        assert results[1].k == 3
        assert MockLda.call_count == 2
        
        # Check LdaModel init calls
        call_args_list = MockLda.call_args_list
        assert call_args_list[0].kwargs["num_topics"] == 2
        assert call_args_list[1].kwargs["num_topics"] == 3


    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_returns_coherence_scores(self, MockCoherence, MockLda, mock_corpus, mock_dictionary):
        """Verify it captures coherence score."""
        mock_coherence = MockCoherence.return_value
        mock_coherence.get_coherence.side_effect = [0.4, 0.6]  # K=2 -> 0.4, K=3 -> 0.6

        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            k_values=[2, 3],
            passes=1
        )

        assert results[0].coherence == 0.4
        assert results[1].coherence == 0.6


# ── train_final_model ─────────────────────────────────────────────────


class TestTrainFinalModel:
    def test_returns_lda_model(self, mock_corpus, mock_dictionary):
        """Integration style test with tiny data/iterations."""
        # actually train a small model (fast)
        model = train_final_model(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            num_topics=2,
            passes=1,
            random_state=42
        )
        assert isinstance(model, LdaModel)
        assert model.num_topics == 2
