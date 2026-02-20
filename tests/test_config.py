"""Tests for pipeline.config — PipelineConfig dataclass."""

from pathlib import Path

import pytest

from pipeline.config import PipelineConfig

# ── Defaults ──────────────────────────────────────────────────────────


class TestPipelineConfigDefaults:
    """Every default declared in the implementation plan must hold."""

    def test_raw_dir(self):
        cfg = PipelineConfig()
        assert cfg.raw_dir == Path("data/raw")

    def test_processed_dir(self):
        cfg = PipelineConfig()
        assert cfg.processed_dir == Path("data/processed")

    def test_included_doc_types(self):
        cfg = PipelineConfig()
        expected = (
            "Conference Paper",
            "Article",
            "Review",
            "Proceedings Paper",
            "Article in Press",
        )
        assert cfg.included_doc_types == expected

    def test_num_passes_sweep(self):
        cfg = PipelineConfig()
        assert cfg.num_passes_sweep == 20

    def test_num_passes_final(self):
        cfg = PipelineConfig()
        assert cfg.num_passes_final == 40

    def test_topic_range(self):
        cfg = PipelineConfig()
        assert cfg.topic_range == (2, 16)

    def test_random_state(self):
        cfg = PipelineConfig()
        assert cfg.random_state == 42

    def test_min_topic_prob(self):
        cfg = PipelineConfig()
        assert cfg.min_topic_prob == pytest.approx(0.7)

    def test_min_citations_default(self):
        cfg = PipelineConfig()
        assert cfg.min_citations == 10

    def test_min_citations_per_topic_default_is_none(self):
        cfg = PipelineConfig()
        assert cfg.min_citations_per_topic is None


# ── Overrides ─────────────────────────────────────────────────────────


class TestPipelineConfigOverrides:
    """Users must be able to override any field at construction time."""

    def test_override_raw_dir(self):
        cfg = PipelineConfig(raw_dir=Path("/tmp/custom"))
        assert cfg.raw_dir == Path("/tmp/custom")

    def test_override_min_citations(self):
        cfg = PipelineConfig(min_citations=5)
        assert cfg.min_citations == 5

    def test_override_min_citations_per_topic(self):
        overrides = {0: 8, 2: 15}
        cfg = PipelineConfig(min_citations_per_topic=overrides)
        assert cfg.min_citations_per_topic == overrides

    def test_override_topic_range(self):
        cfg = PipelineConfig(topic_range=(3, 20))
        assert cfg.topic_range == (3, 20)


# ── Helper: effective_min_citations ───────────────────────────────────


class TestEffectiveMinCitations:
    """Utility method that resolves per-topic overrides with fallback."""

    def test_returns_global_when_no_per_topic(self):
        cfg = PipelineConfig(min_citations=10)
        assert cfg.effective_min_citations(0) == 10
        assert cfg.effective_min_citations(5) == 10

    def test_returns_per_topic_override(self):
        cfg = PipelineConfig(
            min_citations=10,
            min_citations_per_topic={2: 15},
        )
        assert cfg.effective_min_citations(0) == 10  # fallback
        assert cfg.effective_min_citations(2) == 15  # override

    def test_returns_global_for_missing_topic_key(self):
        cfg = PipelineConfig(
            min_citations=10,
            min_citations_per_topic={1: 20},
        )
        assert cfg.effective_min_citations(0) == 10
        assert cfg.effective_min_citations(1) == 20


# ── Recency filter defaults ────────────────────────────────────────────


class TestRecencyDefaults:
    def test_recency_filter_enabled_by_default(self):
        cfg = PipelineConfig()
        assert cfg.recency_filter_enabled is True

    def test_recent_threshold_years_default(self):
        cfg = PipelineConfig()
        assert cfg.recent_threshold_years == 2

    def test_mid_range_threshold_years_default(self):
        cfg = PipelineConfig()
        assert cfg.mid_range_threshold_years == 6

    def test_mid_range_min_citations_default(self):
        cfg = PipelineConfig()
        assert cfg.mid_range_min_citations == 5

    def test_top_n_authors_default(self):
        cfg = PipelineConfig()
        assert cfg.top_n_authors == 15

    def test_top_n_sources_default(self):
        cfg = PipelineConfig()
        assert cfg.top_n_sources == 15

    def test_reference_year_none_by_default(self):
        cfg = PipelineConfig()
        assert cfg.reference_year is None
