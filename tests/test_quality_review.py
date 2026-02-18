"""Tests for pipeline.quality_review — export/import for manual review."""

from pathlib import Path
import pandas as pd
import pytest
from pipeline.quality_review import export_for_review, load_reviewed

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": ["Paper A", "Paper B"],
        "Dominant_Topic": [0, 1],
        "Perc_Contribution": [0.9, 0.8],
        "cited_by": [10, 20]
    })


@pytest.fixture
def review_csv(tmp_path, sample_df):
    """Create a CSV mimicking a Reviewed file."""
    path = tmp_path / "review.csv"
    # Add 'Keep' column as if user edited it
    df = sample_df.copy()
    df["Keep"] = ["y", "n"]  # y=Yes, n=No
    df.to_csv(path, index=False)
    return path


# ── export_for_review ─────────────────────────────────────────────────


class TestExportForReview:
    def test_creates_csv(self, sample_df, tmp_path):
        out_path = tmp_path / "to_review.csv"
        export_for_review(sample_df, out_path)
        
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) == 2
        assert "Keep" in df.columns  # Should add empty column for user
        assert pd.isna(df.iloc[0]["Keep"])  # Should be empty/NaN

    def test_preserves_columns(self, sample_df, tmp_path):
        out_path = tmp_path / "to_review.csv"
        export_for_review(sample_df, out_path)
        df = pd.read_csv(out_path)
        assert "title" in df.columns
        assert "Dominant_Topic" in df.columns


# ── load_reviewed ─────────────────────────────────────────────────────


class TestLoadReviewed:
    def test_loads_and_filters_kept(self, review_csv):
        # review_csv has rows [y, n]
        df = load_reviewed(review_csv)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Paper A"

    def test_handles_different_case(self, tmp_path, sample_df):
        path = tmp_path / "review_case.csv"
        df = sample_df.copy()
        df["Keep"] = ["YES", "No"]  # different case
        df.to_csv(path, index=False)
        
        filtered = load_reviewed(path)
        assert len(filtered) == 1
        assert filtered.iloc[0]["title"] == "Paper A"

    def test_raises_if_keep_column_missing(self, tmp_path, sample_df):
        path = tmp_path / "bad.csv"
        sample_df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Keep"):
            load_reviewed(path)
