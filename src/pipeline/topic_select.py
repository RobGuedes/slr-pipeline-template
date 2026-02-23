"""Step 7 — Topic Selection: filter by probability & citations.

Assigns dominant topic to each document and filters out documents that don't meet
the minimum probability or citation thresholds (globally or per-topic).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gensim.models import LdaModel
    from pipeline.config import PipelineConfig


def assign_dominant_topic(
    df: pd.DataFrame,
    model: LdaModel,
    corpus: list[list[tuple[int, int]]],
) -> pd.DataFrame:
    """Assign dominant topic and its probability to each document.

    Adds "Dominant_Topic" (int) and "Perc_Contribution" (float) columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with documents (aligned with corpus).
    model : LdaModel
        Trained LDA model.
    corpus : list
        BoW corpus corresponding to df rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns.
    """
    df_out = df.copy()

    # Pre-allocate lists for speed
    dom_topics = []
    perc_contribs = []

    # Helper to get topics safely
    def get_topics(bow: list[tuple[int, int]]) -> list[tuple[int, float]]:
        try:
            # minimum_probability=0 ensures all topics are returned?
            # Or use None to get only significant ones.
            # We want the max, so default behavior is usually fine,
            # but if no topics returned (empty doc), we handle it.
            return model.get_document_topics(bow, minimum_probability=0.0)
        except Exception:
            return []

    for bow in corpus:
        topics = get_topics(bow)
        if not topics:
            dom_topics.append(-1)
            perc_contribs.append(0.0)
            continue

        # Sort by probability descending
        topics.sort(key=lambda x: x[1], reverse=True)
        top_topic, top_prob = topics[0]

        dom_topics.append(int(top_topic))
        perc_contribs.append(float(top_prob))

    df_out["Dominant_Topic"] = dom_topics
    df_out["Perc_Contribution"] = perc_contribs

    return df_out


def compute_top_authors(df: pd.DataFrame, n: int = 15) -> list[str]:
    """Return the top *n* authors by paper count from the full dataset.

    Authors are semicolon-separated in the ``author`` column.
    Each author is stripped and counted individually.
    """
    if "author" not in df.columns:
        return []

    authors: list[str] = []
    for raw in df["author"].dropna():
        for name in str(raw).split(";"):
            name = name.strip()
            if name:
                authors.append(name)

    if not authors:
        return []

    counts = pd.Series(authors).value_counts()
    return counts.head(n).index.tolist()


def compute_top_sources(df: pd.DataFrame, n: int = 15) -> list[str]:
    """Return the top *n* publication sources by paper count.

    Uses the ``source_title`` column directly.
    """
    if "source_title" not in df.columns:
        return []

    counts = df["source_title"].dropna().value_counts()
    return counts.head(n).index.tolist()


def recover_recent_papers(
    df: pd.DataFrame,
    config: "PipelineConfig",
    top_authors: list[str],
    top_sources: list[str],
) -> pd.DataFrame:
    """Recover recent papers that were rejected by the strict citation filter.

    Second pass of the two-pass filter. Scans rejected papers and returns
    those meeting age-based recency criteria.

    Age brackets:
    - < recent_threshold_years: 0 citations OK if author or source is relevant
    - recent_threshold_years to < mid_range_threshold_years: >= mid_range_min_citations
    - >= mid_range_threshold_years: not recovered (strict filter handles these)

    Parameters
    ----------
    df : pd.DataFrame
        Rejected papers (those that did NOT pass the strict filter).
    config : PipelineConfig
        Configuration with recency thresholds.
    top_authors : list[str]
        Top N authors by paper count (from full dataset).
    top_sources : list[str]
        Top N publication sources by paper count (from full dataset).

    Returns
    -------
    pd.DataFrame
        Papers recovered by recency criteria.
    """
    if not config.recency_filter_enabled or df.empty:
        return df.iloc[0:0].copy()

    ref_year = (
        config.reference_year
        if config.reference_year is not None
        else datetime.now().year
    )

    # Drop rows with missing year — cannot determine age
    has_year = df["year"].notna()
    df_valid = df[has_year].copy()
    if df_valid.empty:
        return df.iloc[0:0].copy()

    age = ref_year - df_valid["year"].astype(int)

    # Bracket 1: recent (< recent_threshold_years)
    recent_mask = age < config.recent_threshold_years
    if top_authors or top_sources:
        top_authors_set = set(top_authors)
        top_sources_set = set(top_sources)

        def _has_top_author(authors_str: str) -> bool:
            if pd.isna(authors_str):
                return False
            return any(
                a.strip() in top_authors_set for a in str(authors_str).split(";")
            )

        author_relevant = (
            df_valid["author"].apply(_has_top_author)
            if top_authors_set
            else pd.Series(False, index=df_valid.index)
        )
        source_relevant = (
            df_valid["source_title"].isin(top_sources_set)
            if top_sources_set
            else pd.Series(False, index=df_valid.index)
        )
        relevance_mask = author_relevant | source_relevant
    else:
        relevance_mask = pd.Series(False, index=df_valid.index)

    recent_recovered = recent_mask & relevance_mask

    # Bracket 2: mid-range (recent_threshold to < mid_range_threshold)
    mid_mask = (age >= config.recent_threshold_years) & (
        age < config.mid_range_threshold_years
    )
    mid_citation_ok = df_valid["cited_by"] >= config.mid_range_min_citations
    mid_recovered = mid_mask & mid_citation_ok

    # Union
    recovery_mask = recent_recovered | mid_recovered
    return df_valid[recovery_mask].copy()


def filter_documents(
    df: pd.DataFrame,
    config: "PipelineConfig",
    full_df: pd.DataFrame | None = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Filter documents based on probability and citation count.

    Rules:
    1. Perc_Contribution >= config.min_topic_prob
    2. cited_by >= config.effective_min_citations(Dominant_Topic)

    When *full_df* is provided and ``config.recency_filter_enabled`` is True,
    a second recovery pass runs on rejected papers using age-based criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with "Dominant_Topic" and "Perc_Contribution".
    config : PipelineConfig
        Configuration with thresholds.
    full_df : pd.DataFrame | None
        Full dataset (before any filtering) used to compute top-15 authors
        and sources for recency recovery. If None, recovery is skipped.
    return_stats : bool
        If True, returns a tuple of (DataFrame, stats_dict) where stats_dict
        contains filter stage counts.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]
        Filtered DataFrame, or (DataFrame, stats) if return_stats=True.

        The stats dictionary contains:
        - failed_topic_assignment: Count of documents with Dominant_Topic == -1
        - passed_probability: Count meeting min_topic_prob threshold
        - passed_citations: Count meeting min_citations threshold
        - selected_strict: Count passing all strict filters
        - recovered: Count recovered by recency filter (0 if disabled)
    """
    if "Dominant_Topic" not in df.columns or "Perc_Contribution" not in df.columns:
        raise ValueError("DataFrame must have assigned dominant topics.")

    # 1. Filter by probability
    prob_mask = df["Perc_Contribution"] >= config.min_topic_prob

    # 2. Exclude documents that failed topic assignment
    valid_topic_mask = df["Dominant_Topic"] != -1

    # 3. Filter by citations (per-topic adaptive)
    min_citations = df["Dominant_Topic"].map(config.effective_min_citations)
    citation_mask = df["cited_by"] >= min_citations

    # Combine masks
    final_mask = prob_mask & valid_topic_mask & citation_mask

    # Collect stats if requested
    if return_stats:
        stats = {
            "failed_topic_assignment": int((df["Dominant_Topic"] == -1).sum()),
            "passed_probability": int(prob_mask.sum()),
            "passed_citations": int(citation_mask.sum()),
            "selected_strict": int(final_mask.sum()),
            "recovered": 0,
        }

    # 4. Optional second-pass recency recovery
    recovered_df = None
    if config.recency_filter_enabled and full_df is not None:
        top_authors = compute_top_authors(full_df, config.top_n_authors)
        top_sources = compute_top_sources(full_df, config.top_n_sources)
        df_rejected = df[~final_mask]
        recovered_df = recover_recent_papers(df_rejected, config, top_authors, top_sources)
        if return_stats:
            stats["recovered"] = len(recovered_df)
        result = pd.concat([df[final_mask], recovered_df]).copy()
    else:
        result = df[final_mask].copy()

    if return_stats:
        return result, stats
    return result
