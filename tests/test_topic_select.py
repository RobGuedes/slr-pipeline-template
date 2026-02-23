"""Tests for pipeline.topic_select — dominant topic assignment and filtering."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from pipeline.topic_select import (
    assign_dominant_topic,
    filter_documents,
    compute_top_authors,
    compute_top_sources,
    recover_recent_papers,
)
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
        if len(bow) == 1:  # doc 0 (just a hack to distinguish)
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
    return pd.DataFrame({"title": ["Doc A", "Doc B"], "cited_by": [5, 20]})


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
        df = pd.DataFrame(
            {
                "Dominant_Topic": [0, 1],
                "Perc_Contribution": [0.8, 0.5],  # 0.8 > 0.7 (keep), 0.5 < 0.7 (drop)
                "cited_by": [100, 100],
            }
        )
        cfg = PipelineConfig(min_topic_prob=0.7, min_citations=0)
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Dominant_Topic"] == 0

    def test_filters_by_global_citations(self):
        df = pd.DataFrame(
            {
                "Dominant_Topic": [0, 0],
                "Perc_Contribution": [0.9, 0.9],
                "cited_by": [5, 15],  # 5 < 10 (drop), 15 > 10 (keep)
            }
        )
        cfg = PipelineConfig(min_topic_prob=0.0, min_citations=10)
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["cited_by"] == 15

    def test_filters_by_per_topic_citation_override(self):
        df = pd.DataFrame(
            {
                "Dominant_Topic": [0, 1],
                "Perc_Contribution": [0.9, 0.9],
                "cited_by": [12, 12],
            }
        )
        # Topic 0: global min 10 (12 is kept)
        # Topic 1: override min 20 (12 is dropped)
        cfg = PipelineConfig(
            min_topic_prob=0.0, min_citations=10, min_citations_per_topic={1: 20}
        )
        filtered = filter_documents(df, cfg)
        assert len(filtered) == 1
        assert filtered.iloc[0]["Dominant_Topic"] == 0


# ── compute_top_authors ───────────────────────────────────────────────


class TestComputeTopAuthors:
    def test_returns_top_n_by_paper_count(self):
        df = pd.DataFrame(
            {
                "author": [
                    "Smith, J.; Doe, A.",
                    "Smith, J.; Brown, B.",
                    "Doe, A.; Brown, B.",
                    "Smith, J.",
                ]
            }
        )
        top = compute_top_authors(df, n=2)
        # Smith appears in 3 papers, Doe in 2, Brown in 2
        assert top[0] == "Smith, J."
        assert len(top) == 2

    def test_handles_missing_author_column(self):
        df = pd.DataFrame({"title": ["A"]})
        top = compute_top_authors(df, n=5)
        assert top == []

    def test_handles_nan_authors(self):
        df = pd.DataFrame({"author": [None, "Smith, J."]})
        top = compute_top_authors(df, n=5)
        assert "Smith, J." in top


# ── compute_top_sources ───────────────────────────────────────────────


class TestComputeTopSources:
    def test_returns_top_n_by_paper_count(self):
        df = pd.DataFrame(
            {"source_title": ["Journal A", "Journal A", "Journal B", "Journal C"]}
        )
        top = compute_top_sources(df, n=2)
        assert top[0] == "Journal A"
        assert len(top) == 2

    def test_handles_missing_source_column(self):
        df = pd.DataFrame({"title": ["A"]})
        top = compute_top_sources(df, n=5)
        assert top == []

    def test_handles_nan_sources(self):
        df = pd.DataFrame({"source_title": [None, "J1", "J1"]})
        top = compute_top_sources(df, n=5)
        assert "J1" in top


# ── recover_recent_papers ─────────────────────────────────────────────


class TestRecoverRecentPapers:
    """Test the second-pass recency recovery logic."""

    def _make_config(self, **overrides):
        defaults = dict(
            min_topic_prob=0.0,
            min_citations=10,
            recency_filter_enabled=True,
            recent_threshold_years=2,
            mid_range_threshold_years=6,
            mid_range_min_citations=5,
            reference_year=2026,
        )
        defaults.update(overrides)
        return PipelineConfig(**defaults)

    def test_recovers_recent_paper_by_top_author(self):
        """< 2 years, 0 citations, author in top-15 → recovered."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "cited_by": [0],
                "author": ["Smith, J."],
                "source_title": ["Obscure Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        top_authors = ["Smith, J."]
        top_sources = []
        result = recover_recent_papers(
            df, self._make_config(), top_authors, top_sources
        )
        assert len(result) == 1

    def test_recovers_recent_paper_by_top_source(self):
        """< 2 years, 0 citations, source in top-15 → recovered."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "cited_by": [0],
                "author": ["Nobody, X."],
                "source_title": ["Top Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        top_authors = []
        top_sources = ["Top Journal"]
        result = recover_recent_papers(
            df, self._make_config(), top_authors, top_sources
        )
        assert len(result) == 1

    def test_rejects_recent_paper_without_relevance(self):
        """< 2 years, 0 citations, neither top author nor source → not recovered."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "cited_by": [0],
                "author": ["Nobody, X."],
                "source_title": ["Obscure Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 0

    def test_recovers_mid_range_paper_with_enough_citations(self):
        """2-6 years, citations >= mid_range_min → recovered."""
        df = pd.DataFrame(
            {
                "year": [2022],
                "cited_by": [5],
                "author": ["Nobody, X."],
                "source_title": ["Any Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 1

    def test_rejects_mid_range_paper_with_few_citations(self):
        """2-6 years, citations < mid_range_min → not recovered."""
        df = pd.DataFrame(
            {
                "year": [2022],
                "cited_by": [2],
                "author": ["Nobody, X."],
                "source_title": ["Any Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 0

    def test_does_not_recover_old_papers(self):
        """≥ 6 years, even with many citations → not recovered (strict filter handles these)."""
        df = pd.DataFrame(
            {
                "year": [2018],
                "cited_by": [50],
                "author": ["Smith, J."],
                "source_title": ["Top Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(
            df, self._make_config(), ["Smith, J."], ["Top Journal"]
        )
        assert len(result) == 0

    def test_handles_missing_year(self):
        """Papers with no year are not recovered."""
        df = pd.DataFrame(
            {
                "year": [None],
                "cited_by": [0],
                "author": ["Smith, J."],
                "source_title": ["Top Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(
            df, self._make_config(), ["Smith, J."], ["Top Journal"]
        )
        assert len(result) == 0

    def test_multiauthor_match(self):
        """Any author matching top-15 qualifies the paper."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "cited_by": [0],
                "author": ["Nobody, X.; Smith, J.; Other, Y."],
                "source_title": ["Obscure Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        result = recover_recent_papers(df, self._make_config(), ["Smith, J."], [])
        assert len(result) == 1

    def test_disabled_returns_empty(self):
        """When recency_filter_enabled=False, nothing is recovered."""
        df = pd.DataFrame(
            {
                "year": [2025],
                "cited_by": [0],
                "author": ["Smith, J."],
                "source_title": ["Top Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        cfg = self._make_config(recency_filter_enabled=False)
        result = recover_recent_papers(df, cfg, ["Smith, J."], ["Top Journal"])
        assert len(result) == 0

    def test_reference_year_defaults_to_current_year(self):
        """When reference_year is None, uses datetime.now().year."""
        from datetime import datetime

        current_year = datetime.now().year
        df = pd.DataFrame(
            {
                "year": [current_year],
                "cited_by": [0],
                "author": ["Smith, J."],
                "source_title": ["Top Journal"],
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.8],
            }
        )
        cfg = self._make_config(reference_year=None)
        result = recover_recent_papers(df, cfg, ["Smith, J."], ["Top Journal"])
        # age = 0, which is < 2 → recent bracket, top author → recovered
        assert len(result) == 1


# ── filter_documents integration with recovery ────────────────────────


class TestFilterDocumentsWithRecovery:
    """Integration test: strict filter + recency recovery."""

    def test_recovers_recent_paper_rejected_by_strict_filter(self):
        """A recent paper by a top author should survive the full filter."""
        df = pd.DataFrame(
            {
                "Dominant_Topic": [0, 0, 0],
                "Perc_Contribution": [0.9, 0.9, 0.9],
                "cited_by": [50, 0, 3],
                "year": [2015, 2025, 2023],
                "author": ["Old, A.", "Smith, J.", "Mid, M."],
                "source_title": ["J1", "J2", "J3"],
            }
        )
        # full_df has Smith in top authors
        full_df = pd.DataFrame(
            {
                "author": ["Smith, J."] * 20 + ["Other, X."] * 5,
                "source_title": ["J2"] * 20 + ["J3"] * 5,
            }
        )
        cfg = PipelineConfig(
            min_topic_prob=0.0,
            min_citations=10,
            recency_filter_enabled=True,
            reference_year=2026,
            mid_range_min_citations=5,
        )
        result = filter_documents(df, cfg, full_df=full_df)
        # Paper 0: 50 citations, old → passes strict filter
        # Paper 1: 0 citations, recent, top author → recovered
        # Paper 2: 3 citations, mid-range, < 5 → NOT recovered
        assert len(result) == 2
        assert set(result["author"]) == {"Old, A.", "Smith, J."}

    def test_backward_compatible_without_full_df(self):
        """When full_df is not provided, only strict filter applies (backward compat)."""
        df = pd.DataFrame(
            {
                "Dominant_Topic": [0],
                "Perc_Contribution": [0.9],
                "cited_by": [0],
                "year": [2025],
                "author": ["Smith, J."],
                "source_title": ["J1"],
            }
        )
        cfg = PipelineConfig(min_topic_prob=0.0, min_citations=10)
        result = filter_documents(df, cfg)
        assert len(result) == 0  # strict filter rejects, no recovery without full_df


# ── filter_documents with stats ───────────────────────────────────────


class TestFilterDocumentsWithStats:
    def test_returns_filter_stage_counts(self):
        """Verify filter_documents can return intermediate filter counts."""
        config = PipelineConfig(min_topic_prob=0.7, min_citations=5)
        df = pd.DataFrame({
            "Dominant_Topic": [0, 1, 2, -1, 0],
            "Perc_Contribution": [0.8, 0.6, 0.9, 0.0, 0.75],
            "cited_by": [10, 3, 8, 0, 12],
        })
        result, stats = filter_documents(df, config, return_stats=True)
        assert "failed_topic_assignment" in stats
        assert "passed_probability" in stats
        assert "passed_citations" in stats
        assert "selected_strict" in stats
        assert stats["failed_topic_assignment"] == 1  # Dominant_Topic == -1
        # Rows 0,2,4 pass: prob>=0.7, citations>=5, valid topic
        assert stats["selected_strict"] == 3
        assert stats["passed_probability"] == 3  # Rows 0,2,4
        assert stats["passed_citations"] == 3  # Rows 0,2,4
