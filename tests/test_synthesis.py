"""Tests for pipeline.synthesis — reporting and visualization."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pipeline.synthesis import plot_topics, generate_report, _plot_horizontal_bar, plot_bibliometrics


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


# ── plot_bibliometrics ────────────────────────────────────────────────


class TestPlotBibliometrics:
    @pytest.fixture
    def sample_biblio_df(self):
        return pd.DataFrame({
            "year": [2020, 2021, 2022, 2020, 2021],
            "author": [
                "Smith, J.; Doe, A.",
                "Smith, J.",
                "Brown, B.",
                "Doe, A.",
                "Smith, J.; Brown, B.",
            ],
            "source_title": ["J1", "J2", "J1", "J3", "J1"],
            "Affiliations": [
                "Dept A, Uni X, USA",
                "Dept B, Uni Y, UK",
                "Dept C, Uni X, USA",
                None,
                "Dept D, Uni Z, Brazil",
            ],
        })

    def test_creates_all_plot_files(self, sample_biblio_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_bibliometrics(sample_biblio_df, Path(tmpdir))
            assert (Path(tmpdir) / "publication_years.png").exists()
            assert (Path(tmpdir) / "top_authors.png").exists()
            assert (Path(tmpdir) / "top_sources.png").exists()
            assert (Path(tmpdir) / "top_countries.png").exists()
            assert (Path(tmpdir) / "top_affiliations.png").exists()

    def test_no_docs_parameter_required(self, sample_biblio_df):
        """Verify plot_bibliometrics no longer requires a DocumentSet."""
        import inspect
        sig = inspect.signature(plot_bibliometrics)
        param_names = list(sig.parameters.keys())
        assert "docs" not in param_names

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({
            "year": pd.Series(dtype=int),
            "author": pd.Series(dtype=str),
            "source_title": pd.Series(dtype=str),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            plot_bibliometrics(df, Path(tmpdir))
