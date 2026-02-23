"""Pipeline metrics dataclass for synthesis reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline statistics for synthesis reporting.

    Tracks metrics across all pipeline stages: ingestion, topic modeling,
    filtering, and final dataset characteristics.
    """

    # Ingestion
    scopus_raw: int
    wos_raw: int
    duplicates_removed: int
    unique_papers: int

    # Topic modeling
    selected_k: int
    coherence_score: float
    papers_per_topic: dict[int, int]

    # Filtering
    failed_topic_assignment: int
    passed_probability: int
    passed_citations: int
    papers_selected_strict: int
    papers_recovered: int
    papers_final: int

    # Final stats
    year_min: int
    year_max: int
    total_citations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to structured dictionary for JSON export.

        Returns
        -------
        dict
            Nested structure with sections: ingestion, topic_modeling,
            filtering, final_dataset.
        """
        return {
            "ingestion": {
                "scopus_raw": self.scopus_raw,
                "wos_raw": self.wos_raw,
                "duplicates_removed": self.duplicates_removed,
                "unique_papers": self.unique_papers,
            },
            "topic_modeling": {
                "selected_k": self.selected_k,
                "coherence_score": self.coherence_score,
                "papers_per_topic": self.papers_per_topic,
            },
            "filtering": {
                "failed_topic_assignment": self.failed_topic_assignment,
                "passed_probability": self.passed_probability,
                "passed_citations": self.passed_citations,
                "papers_selected_strict": self.papers_selected_strict,
                "papers_recovered": self.papers_recovered,
                "papers_final": self.papers_final,
            },
            "final_dataset": {
                "year_min": self.year_min,
                "year_max": self.year_max,
                "total_citations": self.total_citations,
            },
        }
