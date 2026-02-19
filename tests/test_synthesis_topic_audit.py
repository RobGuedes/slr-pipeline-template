"""Tests for pipeline.synthesis.plot_topic_audit — topic count audit plot."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pipeline.topic_model import SweepResult
from pipeline.synthesis import plot_topic_audit


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_sweep_results():
    """Create a list of dummy SweepResult objects for testing."""
    return [
        SweepResult(k=2, coherence=0.35, perplexity=-7.0),
        SweepResult(k=3, coherence=0.42, perplexity=-6.5),
        SweepResult(k=4, coherence=0.50, perplexity=-6.8),
        SweepResult(k=5, coherence=0.48, perplexity=-6.4),
    ]


# ── Tests ─────────────────────────────────────────────────────────────


class TestPlotTopicAudit:
    def test_creates_png_file(self, mock_sweep_results):
        """Verify the function creates a non-empty PNG file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "audit.png"
            plot_topic_audit(mock_sweep_results, out_path)

            assert out_path.exists(), "Audit plot PNG was not created"
            assert out_path.stat().st_size > 0, "Audit plot PNG is empty"

    def test_creates_parent_dirs(self, mock_sweep_results):
        """Verify the function creates parent directories if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "nested" / "dir" / "audit.png"
            plot_topic_audit(mock_sweep_results, out_path)

            assert out_path.exists()
