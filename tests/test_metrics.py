"""Tests for pipeline.metrics — pipeline statistics tracking."""

from pipeline.metrics import PipelineMetrics


class TestPipelineMetrics:
    def test_creates_instance_with_all_fields(self):
        metrics = PipelineMetrics(
            scopus_raw=450,
            wos_raw=320,
            duplicates_removed=85,
            unique_papers=685,
            selected_k=8,
            coherence_score=0.542,
            papers_per_topic={0: 85, 1: 72},
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
        assert metrics.scopus_raw == 450
        assert metrics.selected_k == 8
        assert metrics.papers_per_topic[0] == 85

    def test_to_dict_converts_to_structured_format(self):
        metrics = PipelineMetrics(
            scopus_raw=100, wos_raw=50, duplicates_removed=10, unique_papers=140,
            selected_k=5, coherence_score=0.5, papers_per_topic={0: 20},
            failed_topic_assignment=5, passed_probability=100, passed_citations=90,
            papers_selected_strict=80, papers_recovered=10, papers_final=90,
            year_min=2020, year_max=2025, total_citations=1000,
        )
        d = metrics.to_dict()
        assert d["ingestion"]["scopus_raw"] == 100
        assert d["topic_modeling"]["selected_k"] == 5
        assert d["filtering"]["papers_final"] == 90
