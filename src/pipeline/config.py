"""Central configuration for the SLR pipeline.

All constants previously scattered across ``globalVar.py`` and notebook
hard-coded values are consolidated here in one ``@dataclass``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Immutable-ish bag of every tuneable knob in the pipeline.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw Scopus CSV / WoS TXT exports.
    processed_dir : Path
        Directory for pipeline outputs (CSVs, models, charts).
    included_doc_types : tuple[str, ...]
        Document types to keep during ingestion (from ``globalVar.INCLUDED_TYPES``).
    num_passes_sweep : int
        Gensim LDA passes during the sweep phase (Step 5).
    num_passes_final : int
        Gensim LDA passes for the final model (Step 6).
    topic_range : tuple[int, int]
        ``range(start, stop)`` of *K* values to sweep (Step 5).
    random_state : int
        Seed for reproducibility.
    min_topic_prob : float
        Minimum document-topic probability to keep (Step 7).
    min_citations : int
        Global default minimum citation count for topic selection (Step 7).
    min_citations_per_topic : dict[int, int] | None
        Per-topic overrides for ``min_citations``, e.g. ``{2: 15}``.
    """

    # ── Paths ──────────────────────────────────────────────────────
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # ── Document-type filter (from globalVar.INCLUDED_TYPES) ──────
    included_doc_types: tuple[str, ...] = (
        "Conference Paper",
        "Article",
        "Review",
        "Proceedings Paper",
        "Article in Press",
    )

    # ── LDA sweep defaults (from notebook cell 15) ────────────────
    num_passes_sweep: int = 20
    num_passes_final: int = 40
    topic_range: tuple[int, int] = (2, 16)
    random_state: int = 42

    # ── Selection thresholds (from notebook cells 22-24) ──────────
    min_topic_prob: float = 0.7
    min_citations: int = 10
    min_citations_per_topic: dict[int, int] | None = field(default=None)

    # ── Helpers ────────────────────────────────────────────────────

    def effective_min_citations(self, topic_id: int) -> int:
        """Return the citation threshold for *topic_id*.

        Checks ``min_citations_per_topic`` first; falls back to
        the global ``min_citations`` default.
        """
        if self.min_citations_per_topic is not None:
            return self.min_citations_per_topic.get(topic_id, self.min_citations)
        return self.min_citations
