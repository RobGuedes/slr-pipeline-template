"""Tests for pipeline.synthesis — reporting and visualization."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pipeline.synthesis import convert_to_litstudy, plot_topics, generate_report, _plot_horizontal_bar


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": ["Paper A", "Paper B"],
        "author": ["Smith, J.", "Doe, A."],
        "year": [2020, 2021],
        "source_title": ["Journal X", "Journal Y"],
        "doi": ["10.1000/1", "10.1000/2"],
        "cited_by": [10, 20],
        "abstract": ["Abstract A", "Abstract B"],
        # Extra columns needed for LitStudy DocumentSet
        "publisher": ["Pub A", "Pub B"],
        "affiliation": ["Aff A", "Aff B"]
    })


# ── _plot_horizontal_bar ──────────────────────────────────────────────


class TestPlotHorizontalBar:
    def test_creates_png_file(self):
        data = pd.Series({"A": 10, "B": 5, "C": 3})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.png"
            _plot_horizontal_bar(data, "No. of documents", "Test Title", out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_creates_parent_dirs(self):
        data = pd.Series({"A": 10})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dir" / "test.png"
            _plot_horizontal_bar(data, "No. of documents", "Test Title", out)
            assert out.exists()


# ── convert_to_litstudy ───────────────────────────────────────────────


class TestConvertToLitStudy:
    def test_returns_document_set(self, sample_df):
        # We need to mock litstudy.DocumentSet or litstudy.load_pandas
        # Since we use pandas internally, we'll try to rely on litstudy's real
        # object if installed, or mock it if dependencies are heavy.
        # Patch litstudy module imported in synthesis.py
        with patch("pipeline.synthesis.litstudy") as MockLitStudy:
            docs = convert_to_litstudy(sample_df)
            # Verify load_csv was called
            # Since we use temp file, we just check called
            assert MockLitStudy.load_csv.called

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"title": ["A"]})
        with patch("pipeline.synthesis.litstudy") as MockLitStudy:
            convert_to_litstudy(df)
            assert MockLitStudy.load_csv.called


# ── generate_report ───────────────────────────────────────────────────


class TestGenerateReport:
    def test_returns_stats_dict(self, sample_df):
        stats = generate_report(sample_df)
        assert stats["total_papers"] == 2
        assert stats["min_year"] == 2020
        assert stats["max_year"] == 2021
        assert stats["total_citations"] == 30


# ── plot_topics ───────────────────────────────────────────────────────


class TestPlotTopics:
    @patch("pipeline.synthesis.pyLDAvis.gensim_models.prepare")
    @patch("pipeline.synthesis.pyLDAvis.save_html")
    def test_calls_pyldavis(self, mock_save, mock_prepare):
        model = MagicMock()
        corpus = MagicMock()
        dictionary = MagicMock()
        
        plot_topics(model, corpus, dictionary, "output.html")
        
        mock_prepare.assert_called_once()
        mock_save.assert_called_once()
