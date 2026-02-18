"""Tests for pipeline.ingest — load, merge, de-duplicate."""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import PipelineConfig
from pipeline.ingest import (
    deduplicate,
    ingest_all,
    load_scopus_csv,
    load_wos_txt,
    merge_sources,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── load_scopus_csv ───────────────────────────────────────────────────


class TestLoadScopusCsv:
    def test_returns_dataframe(self):
        df = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        for col in ("author", "title", "year", "doi", "abstract",
                     "cited_by", "document_type", "source_db"):
            assert col in df.columns, f"missing column: {col}"

    def test_source_db_is_scopus(self):
        df = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        assert (df["source_db"] == "Scopus").all()

    def test_row_count(self):
        df = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        assert len(df) == 4  # all 4 rows before filtering/dedup

    def test_cited_by_is_numeric(self):
        df = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        assert pd.api.types.is_numeric_dtype(df["cited_by"])


# ── load_wos_txt ──────────────────────────────────────────────────────


class TestLoadWosTxt:
    def test_returns_dataframe(self):
        df = load_wos_txt(FIXTURES / "wos_sample.txt")
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        df = load_wos_txt(FIXTURES / "wos_sample.txt")
        for col in ("author", "title", "year", "doi", "abstract",
                     "cited_by", "document_type", "source_db"):
            assert col in df.columns, f"missing column: {col}"

    def test_source_db_is_wos(self):
        df = load_wos_txt(FIXTURES / "wos_sample.txt")
        assert (df["source_db"] == "WoS").all()

    def test_row_count(self):
        df = load_wos_txt(FIXTURES / "wos_sample.txt")
        assert len(df) == 3


# ── merge_sources ─────────────────────────────────────────────────────


class TestMergeSources:
    def test_concat_two_sources(self):
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        w = load_wos_txt(FIXTURES / "wos_sample.txt")
        merged = merge_sources(s, w)
        assert len(merged) == len(s) + len(w)

    def test_has_both_source_dbs(self):
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        w = load_wos_txt(FIXTURES / "wos_sample.txt")
        merged = merge_sources(s, w)
        assert set(merged["source_db"].unique()) == {"Scopus", "WoS"}


# ── deduplicate ───────────────────────────────────────────────────────


class TestDeduplicate:
    def test_removes_exact_doi_duplicates(self):
        """Scopus fixture has two rows with DOI 10.1000/test.001."""
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        deduped = deduplicate(s)
        doi_count = (deduped["doi"] == "10.1000/test.001").sum()
        assert doi_count == 1

    def test_removes_cross_source_doi_duplicates(self):
        """Scopus + WoS share DOI 10.1000/test.001."""
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        w = load_wos_txt(FIXTURES / "wos_sample.txt")
        merged = merge_sources(s, w)
        deduped = deduplicate(merged)
        doi_count = (deduped["doi"] == "10.1000/test.001").sum()
        assert doi_count == 1

    def test_averages_citations_on_dedup(self):
        """When de-duplicating, cited_by should be averaged."""
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        deduped = deduplicate(s)
        row = deduped[deduped["doi"] == "10.1000/test.001"].iloc[0]
        # Original: 15 and 18 → average ≈ 16 (int)
        assert row["cited_by"] in (16, 16.5)  # depending on int/float

    def test_keeps_unique_papers(self):
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        deduped = deduplicate(s)
        assert (deduped["doi"] == "10.1000/test.002").sum() == 1

    def test_preserves_non_duplicate_count(self):
        """4 rows - 1 dup DOI = 3 unique papers."""
        s = load_scopus_csv(FIXTURES / "scopus_sample.csv")
        deduped = deduplicate(s)
        assert len(deduped) == 3


# ── ingest_all ────────────────────────────────────────────────────────


class TestIngestAll:
    def test_filters_document_types(self):
        """'Note' in fixture should be excluded by default config."""
        cfg = PipelineConfig(raw_dir=FIXTURES)
        df = ingest_all(cfg.raw_dir, cfg)
        assert "Note" not in df["document_type"].values

    def test_deduplicates_across_sources(self):
        cfg = PipelineConfig(raw_dir=FIXTURES)
        df = ingest_all(cfg.raw_dir, cfg)
        doi_count = (df["doi"] == "10.1000/test.001").sum()
        assert doi_count == 1

    def test_returns_nonempty(self):
        cfg = PipelineConfig(raw_dir=FIXTURES)
        df = ingest_all(cfg.raw_dir, cfg)
        assert len(df) > 0
