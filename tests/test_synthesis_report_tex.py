"""Tests for pipeline.synthesis.export_report_tex — LaTeX report generation."""

import tempfile
from pathlib import Path

import pytest

from pipeline.synthesis import export_report_tex


def test_export_report_tex_creates_file():
    """Verify that export_report_tex creates a non-empty .tex file with expected content."""
    stats = {
        "total_papers": 100,
        "min_year": 2010,
        "max_year": 2024,
        "total_citations": 500
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.tex"
        export_report_tex(stats, out_path)

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        
        # Verify LaTeX specific elements
        assert r"\begin{table}" in content
        assert r"\end{table}" in content
        assert r"Total Documents & 100" in content
        assert r"2010 -- 2024" in content
        assert r"Total Citations & 500" in content


def test_export_report_tex_handles_empty_stats():
    """Verify that export_report_tex handles missing or empty stats gracefully."""
    stats = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "empty_report.tex"
        export_report_tex(stats, out_path)

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "Total Documents & 0" in content
        assert "Year Range & N/A" in content
