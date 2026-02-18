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

    # LitStudy's load_pandas expects a DataFrame where columns match specific names
    # or it tries to guess.
    # Our columns: author, title, year, source_title, doi, cited_by, abstract
    # LitStudy useful columns: Authors, Title, Year, Source title, DOI, Cited by
    
    # We create a copy with "LitStudy-friendly" column names if needed.
    # Actually, litstudy.load_pandas is flexible.
    # Let's try passing directly, but ensuring keys exist.
    
    # We can also construct DocumentSet directly from a list of dicts if needed,
    # but load_pandas is idiomatic.
    
    # Helper to format authors list string "Smith, J.; Doe, A." -> ["Smith, J.", "Doe, A."]
    # Our 'author' column from ingest is just the raw string (e.g. "Smith, J.; Doe, A.").
    # LitStudy parses this string if we tell it the column name.
    
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
    
    # Pass to LitStudy
    # It might warn about missing columns, but basic plots should work.
    docs = litstudy.load_pandas(df_lit)
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
