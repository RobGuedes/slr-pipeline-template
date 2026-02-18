"""Tests for pipeline.topic_select — dominant topic assignment and filtering."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from pipeline.topic_select import assign_dominant_topic, filter_documents
from pipeline.config import PipelineConfig

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_lda_model():
    """Mock LdaModel.get_document_topics."""
    model = MagicMock()
    # Mock document-topic distribution
    # doc 0: topic 0 (0.9), topic 1 (0.1)
    # doc 1: topic 0 (0.4), topic 1 (0.6)
    def side_effect(bow, minimum_probability=None):
        if len(bow) == 1: # doc 0 (just a hack to distinguish)
            return [(0, 0.9), (1, 0.1)]
        else:
            return [(0, 0.4), (1, 0.6)]
    
    model.get_document_topics.side_effect = side_effect
    return model


@pytest.fixture
def mock_corpus():
    # doc 0 has 1 word, doc 1 has 2 words (to distinguish in mock side_effect)
    return [[(0, 1)], [(0, 1), (1, 1)]]


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": ["Doc A", "Doc B"],
        "cited_by": [5, 20]
    })


# ── assign_dominant_topic ─────────────────────────────────────────────


class TestAssignDominantTopic:
    def test_adds_dominant_columns(self, mock_lda_model, mock_corpus, sample_df):
        # We also need 'texts' but assign_dominant_topic usually takes df + corpus + model
        df = assign_dominant_topic(sample_df, mock_lda_model, mock_corpus)
        
        assert "Dominant_Topic" in df.columns
        assert "Perc_Contribution" in df.columns
        
        # Doc 0 -> Topic 0 (0.9)
        assert df.loc[0, "Dominant_Topic"] == 0
        assert df.loc[0, "Perc_Contribution"] == 0.9
        
        # Doc 1 -> Topic 1 (0.6)
        assert df.loc[1, "Dominant_Topic"] == 1
        assert df.loc[1, "Perc_Contribution"] == 0.6


# ── filter_documents ──────────────────────────────────────────────────


class TestFilterDocuments:
    def test_filters_by_probability(self):
        df = pd.DataFrame({
            "Dominant_Topic": [0, 1],
            "Perc_Contribution": [0.8, 0.5],  # 0.8 > 0.7 (keep), 0.5 < 0.7 (drop)
            "cited_by": [100, 100]
        })
        cfg = PipelineConfig(min_topic_prob=0.7, min_citations=0)
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Dominant_Topic"] == 0

    def test_filters_by_global_citations(self):
        df = pd.DataFrame({
            "Dominant_Topic": [0, 0],
            "Perc_Contribution": [0.9, 0.9],
            "cited_by": [5, 15]  # 5 < 10 (drop), 15 > 10 (keep)
        })
        cfg = PipelineConfig(min_topic_prob=0.0, min_citations=10)
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["cited_by"] == 15

    def test_filters_by_per_topic_citation_override(self):
        df = pd.DataFrame({
            "Dominant_Topic": [0, 1],
            "Perc_Contribution": [0.9, 0.9],
            "cited_by": [12, 12]
        })
        # Topic 0: global min 10 (12 is kept)
        # Topic 1: override min 20 (12 is dropped)
        cfg = PipelineConfig(
            min_topic_prob=0.0,
            min_citations=10,
            min_citations_per_topic={1: 20}
        )
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Dominant_Topic"] == 0
