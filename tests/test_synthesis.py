"""Tests for pipeline.synthesis — reporting and visualization."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from pipeline.synthesis import (
    plot_topics,
    generate_report,
    export_report_tex,
    export_report_json,
    _plot_horizontal_bar,
    plot_bibliometrics,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "title": ["Paper A", "Paper B"],
            "author": ["Smith, J.", "Doe, A."],
            "year": [2020, 2021],
            "source_title": ["Journal X", "Journal Y"],
            "doi": ["10.1000/1", "10.1000/2"],
            "cited_by": [10, 20],
            "abstract": ["Abstract A", "Abstract B"],
            # Extra columns needed for LitStudy DocumentSet
            "publisher": ["Pub A", "Pub B"],
            "affiliation": ["Aff A", "Aff B"],
        }
    )


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
        return pd.DataFrame(
            {
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
                    "Department of Economics, Uni X, CityA, StateA, USA",
                    "Uni Y, London, UK",
                    "School of Business, Uni X, CityA, StateA, USA",
                    None,
                    "Uni Z, São Paulo, SP, Brazil",
                ],
            }
        )

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
        df = pd.DataFrame(
            {
                "year": pd.Series(dtype=int),
                "author": pd.Series(dtype=str),
                "source_title": pd.Series(dtype=str),
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            plot_bibliometrics(df, Path(tmpdir))


# ── generate_report with PipelineMetrics ──────────────────────────


class TestGenerateReportWithMetrics:
    def test_accepts_pipeline_metrics(self):
        """Verify generate_report works with PipelineMetrics."""
        from pipeline.metrics import PipelineMetrics

        metrics = PipelineMetrics(
            scopus_raw=100,
            wos_raw=50,
            duplicates_removed=10,
            unique_papers=140,
            selected_k=5,
            coherence_score=0.5,
            papers_per_topic={0: 20},
            failed_topic_assignment=5,
            passed_probability=100,
            passed_citations=90,
            papers_selected_strict=80,
            papers_recovered=10,
            papers_final=90,
            year_min=2020,
            year_max=2025,
            total_citations=1000,
        )
        stats = generate_report(metrics)
        assert stats["ingestion"]["scopus_raw"] == 100
        assert stats["topic_modeling"]["selected_k"] == 5
        assert stats["filtering"]["papers_final"] == 90


# ── export_report_tex with PipelineMetrics ────────────────────────


class TestExportReportTexWithMetrics:
    def test_exports_comprehensive_latex_table(self):
        """Verify export_report_tex formats PipelineMetrics as LaTeX."""
        from pipeline.metrics import PipelineMetrics

        metrics = PipelineMetrics(
            scopus_raw=450,
            wos_raw=320,
            duplicates_removed=85,
            unique_papers=685,
            selected_k=8,
            coherence_score=0.542,
            papers_per_topic={},
            failed_topic_assignment=12,
            passed_probability=520,
            passed_citations=420,
            papers_selected_strict=380,
            papers_recovered=35,
            papers_final=415,
            year_min=2016,
            year_max=2026,
            total_citations=7254,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.tex"
            export_report_tex(metrics, out)
            assert out.exists()
            content = out.read_text()
            assert "Scopus (raw)" in content
            assert "450" in content
            assert "Topics (K)" in content
            assert "8" in content


# ── export_report_json ────────────────────────────────────────────


class TestExportReportJson:
    def test_exports_json_metadata(self):
        """Verify export_report_json writes structured JSON."""
        from pipeline.metrics import PipelineMetrics
        import json

        metrics = PipelineMetrics(
            scopus_raw=100,
            wos_raw=50,
            duplicates_removed=10,
            unique_papers=140,
            selected_k=5,
            coherence_score=0.5,
            papers_per_topic={0: 20, 1: 30},
            failed_topic_assignment=5,
            passed_probability=100,
            passed_citations=90,
            papers_selected_strict=80,
            papers_recovered=10,
            papers_final=90,
            year_min=2020,
            year_max=2025,
            total_citations=1000,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "metadata.json"
            export_report_json(metrics, out)
            assert out.exists()
            with open(out) as f:
                data = json.load(f)
            assert data["ingestion"]["scopus_raw"] == 100
            assert data["topic_modeling"]["papers_per_topic"]["0"] == 20
            assert data["filtering"]["papers_final"] == 90
