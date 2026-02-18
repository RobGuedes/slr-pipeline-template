"""Step 9 — Synthesis: visualized analysis and reporting.

Leverages LitStudy for bibliometric plots and pyLDAvis for interactive topic maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import litstudy

if TYPE_CHECKING:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel
    from litstudy import DocumentSet


def convert_to_litstudy(df: pd.DataFrame) -> "DocumentSet":
    """Convert the pipeline DataFrame to a LitStudy DocumentSet for plotting.

    LitStudy expects specific columns. We map our canonical columns to what
    LitStudy's `load_pandas` or internal structure expects.
    """
    import tempfile
    import os

    # LitStudy doesn't support load_pandas directly.
    # We save to a temporary CSV and load it back using load_csv.
    
    # Rename to standard bibliometric fields usually works best
    mapping = {
        "author": "Authors",
        "title": "Title",
        "year": "Year",
        "source_title": "Source title",
        "doi": "DOI",
        "cited_by": "Cited by",
        "abstract": "Abstract"
    }
    
    # Create mapped dataframe
    df_lit = df.rename(columns=mapping)
    
    # Use temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df_lit.to_csv(tmp, index=False)
        tmp_path = tmp.name
        
    try:
        # load_csv is generic
        docs = litstudy.load_csv(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return docs


def generate_report(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate summary statistics for the final dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataset.

    Returns
    -------
    dict
        Dictionary of statistics (total papers, year range, total citations).
    """
    if len(df) == 0:
        return {
            "total_papers": 0,
            "min_year": None,
            "max_year": None,
            "total_citations": 0
        }

    return {
        "total_papers": len(df),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "total_citations": int(df["cited_by"].sum()),
    }


def plot_topics(
    model: LdaModel,
    corpus: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    output_path: Path | str
) -> None:
    """Generate interactive pyLDAvis plot and save to HTML.

    Parameters
    ----------
    model : LdaModel
        Trained LDA model.
    corpus : list
        Gensim corpus.
    dictionary : Dictionary
        Gensim dictionary.
    output_path : Path | str
        Path to save the HTML file.
    """
    # Prepare the visualization data
    # sort_topics=False preserves the model's topic IDs
    vis_data = pyLDAvis.gensim_models.prepare(
        model, corpus, dictionary, sort_topics=False
    )
    
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to HTML
    pyLDAvis.save_html(vis_data, str(output_path))
