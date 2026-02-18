"""Step 7 — Topic Selection: filter by probability & citations.

Assigns dominant topic to each document and filters out documents that don't meet
the minimum probability or citation thresholds (globally or per-topic).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def filter_documents(
    df: pd.DataFrame,
    config: "PipelineConfig"
) -> pd.DataFrame:
    """Filter documents based on probability and citation count.

    Rules:
    1. Perc_Contribution >= config.min_topic_prob
    2. cited_by >= config.effective_min_citations(Dominant_Topic)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with "Dominant_Topic" and "Perc_Contribution".
    config : PipelineConfig
        Configuration with thresholds.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if "Dominant_Topic" not in df.columns or "Perc_Contribution" not in df.columns:
        raise ValueError("DataFrame must have assigned dominant topics.")

    # 1. Filter by probability
    # Using mask for efficiency
    prob_mask = df["Perc_Contribution"] >= config.min_topic_prob
    
    # 2. Filter by citations (per-topic adaptive)
    # Vectorized approach is hard with per-topic config.
    # Apply row-wise or simple map. Map is efficient.
    
    # Create a map of topic -> min_citations
    # We don't know all topics in advance without model, but we know present topics
    present_topics = df["Dominant_Topic"].unique()
    
    # If topic is -1 (failed), threshold doesn't matter, it's garbage.
    # We should filter out -1 first?
    valid_topic_mask = df["Dominant_Topic"] != -1
    
    # Calculate thresholds for each row
    # This is fast enough for <10k rows.
    # df.apply is slow, List comprehension is faster.
    
    min_citations = df["Dominant_Topic"].map(config.effective_min_citations)
    citation_mask = df["cited_by"] >= min_citations
    
    # Combine masks
    final_mask = prob_mask & valid_topic_mask & citation_mask
    
    return df[final_mask].copy()
