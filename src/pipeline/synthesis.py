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


def plot_bibliometrics(
    df: pd.DataFrame,
    docs: "DocumentSet",
    output_dir: Path | str,
) -> None:
    """Generate bibliometric plots and save them to output_dir.

    Uses LitStudy for year/author histograms (which parse correctly from CSV),
    and parses countries/affiliations directly from the DataFrame's
    ``Affiliations`` column (semicolon-separated entries where the last
    comma-separated token is typically the country).

    Plots generated:
    - publication_years.png
    - top_authors.png
    - top_countries.png
    - top_affiliations.png
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Publication Years (via LitStudy) ────────────────────────
    plt.figure(figsize=(10, 5))
    litstudy.plot_year_histogram(docs)
    plt.title("Publication Trends")
    plt.tight_layout()
    plt.savefig(out / "publication_years.png", dpi=150)
    plt.close()

    # ── 2. Top Authors (via LitStudy) ──────────────────────────────
    plt.figure(figsize=(10, 5))
    litstudy.plot_author_histogram(docs, limit=20)
    plt.title("Top 20 Authors")
    plt.tight_layout()
    plt.savefig(out / "top_authors.png", dpi=150)
    plt.close()

    # ── 3. Top Countries (parsed from DataFrame) ──────────────────
    aff_col = "Affiliations" if "Affiliations" in df.columns else "affiliations"
    if aff_col in df.columns:
        countries: list[str] = []
        for raw in df[aff_col].dropna():
            # Each entry is semicolon-separated; last token after comma is country
            for entry in str(raw).split(";"):
                parts = [p.strip() for p in entry.split(",")]
                if parts:
                    countries.append(parts[-1])

        if countries:
            country_counts = pd.Series(countries).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            country_counts.plot.barh(ax=ax, color="#4c72b0")
            ax.set_xlabel("Number of Affiliations")
            ax.set_title("Top 20 Countries")
            ax.invert_yaxis()
            plt.tight_layout()
            fig.savefig(out / "top_countries.png", dpi=150)
            plt.close(fig)

    # ── 4. Top Affiliations / Institutions (parsed from DataFrame) ─
    if aff_col in df.columns:
        institutions: list[str] = []
        for raw in df[aff_col].dropna():
            for entry in str(raw).split(";"):
                inst = entry.strip().split(",")[0].strip()
                if inst:
                    institutions.append(inst)

        if institutions:
            inst_counts = pd.Series(institutions).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            inst_counts.plot.barh(ax=ax, color="#55a868")
            ax.set_xlabel("Number of Entries")
            ax.set_title("Top 20 Affiliations")
            ax.invert_yaxis()
            plt.tight_layout()
            fig.savefig(out / "top_affiliations.png", dpi=150)
            plt.close(fig)
