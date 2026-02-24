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
    recency_filter_enabled : bool
        Enable age-aware citation recovery in Step 7 (default: True).
    recent_threshold_years : int
        Papers younger than this (in years) qualify for the recent bracket.
    mid_range_threshold_years : int
        Papers between recent_threshold_years and this qualify for mid-range
        recovery if they have enough citations.
    mid_range_min_citations : int
        Minimum citations required for mid-range recovery.
    top_n_authors : int
        Number of top authors (by paper count) used for recency recovery.
    top_n_sources : int
        Number of top sources (by paper count) used for recency recovery.
    reference_year : int | None
        Year used as "now" when computing paper age. Defaults to current year.
    academic_stopwords : tuple[str, ...]
        Common academic terms removed during preprocessing. These words
        appear in most papers regardless of topic and add noise to LDA.
    domain_stopwords : tuple[str, ...]
        User-supplied domain-specific stopwords (empty by default).
        Set per-project for terms ubiquitous in your field.
    nouns_only : bool
        If True, keep only nouns after POS tagging (default: False).
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

    # ── Recency-aware citation recovery (Step 7) ────────────────
    recency_filter_enabled: bool = True
    recent_threshold_years: int = 2
    mid_range_threshold_years: int = 6
    mid_range_min_citations: int = 5
    top_n_authors: int = 15
    top_n_sources: int = 15
    reference_year: int | None = None

    # ── Preprocessing (Step 4) ────────────────────────────────────
    academic_stopwords: tuple[str, ...] = (
        "study",
        "research",
        "result",
        "method",
        "approach",
        "analysis",
        "finding",
        "paper",
        "propose",
        "investigate",
        "examine",
        "evaluate",
        "demonstrate",
        "parameter",
        "present",
        "discuss",
        "conclude",
        "suggest",
        "indicate",
        "reveal",
        "aim",
        "also",
        "objective",
        "contribution",
        "limitation",
        "implication",
        "hypothesis",
        "conclusion",
        "introduction",
        "literature",
        "review",
        "methodology",
        "framework",
        "significant",
        "significantly",
        "respectively",
        "furthermore",
        "moreover",
        "however",
        "therefore",
        "consequently",
        "nevertheless",
        "whereas",
        "although",
        "author",
        "application",
        "capture",
        "process",
        "performance",
        "simulation",
        "accuracy",
        "compare",
        "fit",
        "different",
        "sample",
        "algorithm",
        "evidence",
        "mean",
        "outperform",
        "show",
        "compare",
        "estimate",
        "application",
        "include",
        "apply",
        "well",
        "develop",
        "find",
        "provide",
        "empirical",
        "elsevier",
        "period",
        "information",
        "component",
        "effect",
        "year",
        "john",
        "wiley",
        "son",
    )
    domain_stopwords: tuple[str, ...] = (
        "market", 
        "data",
        "financial",
        "model",
        "forecast", 
        "factor", 
        "risk", 
        "curve", 
        "structure", 
        "time", 
        "return", 
        "interest", 
        "policy", 
        "change",
        "yield", 
        "rate", 
        "bond", 
        "price", 
        "asset", 
        "term", 
        "prediction",
        "predict",
        "predictive",
        "test",
        "high",
        "low",
        "long",
        "short",
        "series",
        "two",
        "three",
    )
    nouns_only: bool = True

    # ── Helpers ────────────────────────────────────────────────────

    def effective_min_citations(self, topic_id: int) -> int:
        """Return the citation threshold for *topic_id*.

        Checks ``min_citations_per_topic`` first; falls back to
        the global ``min_citations`` default.
        """
        if self.min_citations_per_topic is not None:
            return self.min_citations_per_topic.get(topic_id, self.min_citations)
        return self.min_citations
