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


def export_report_tex(stats: dict[str, Any], output_path: Path | str) -> None:
    """Format the synthesis statistics as a LaTeX table and save to .tex file.

    Parameters
    ----------
    stats : dict[str, Any]
        Dictionary of statistics from generate_report.
    output_path : Path | str
        Path to save the .tex file.
    """
    total_docs = stats.get("total_papers", 0)
    min_year = stats.get("min_year", "")
    max_year = stats.get("max_year", "")
    total_cites = stats.get("total_citations", 0)

    year_range = f"{min_year} -- {max_year}" if min_year and max_year else "N/A"

    tex_content = rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lr}}
\hline
\textbf{{Metric}} & \textbf{{Value}} \\
\hline
Total Documents & {total_docs} \\
Year Range & {year_range} \\
Total Citations & {total_cites} \\
\hline
\end{{tabular}}
\caption{{Synthesis Report Statistics}}
\label{{tab:synthesis_report}}
\end{{table}}
"""

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex_content)


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


def plot_topic_audit(
    sweep_results: list,
    output_path: Path | str,
) -> None:
    """Generate a dual-panel audit plot showing Perplexity and Coherence vs K.

    This mirrors the legacy notebook's approach: side-by-side charts of
    Perplexity (blue) and Coherence (orange) across the range of tested
    topic counts, allowing the user to visually evaluate the optimal K.

    Parameters
    ----------
    sweep_results : list[SweepResult]
        Results from ``perform_lda_sweep``.  Each must have ``k``,
        ``perplexity`` and ``coherence`` attributes.
    output_path : Path | str
        Where to save the PNG chart.
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt

    k_values      = [r.k for r in sweep_results]
    perplexities  = [r.perplexity for r in sweep_results]
    coherences    = [r.coherence for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: Perplexity ─────────────────────────────────────
    ax1.plot(k_values, perplexities, marker="o", color="#4c72b0")
    ax1.set_title("Perplexity by Number of Topics")
    ax1.set_xlabel("Number of topics")
    ax1.set_ylabel("Log Perplexity")

    # ── Right panel: Coherence ─────────────────────────────────────
    ax2.plot(k_values, coherences, marker="o", color="orange")
    ax2.set_title("Coherence by Number of Topics")
    ax2.set_xlabel("Number of topics")
    ax2.set_ylabel("Coherence")

    plt.tight_layout()

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


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
            ax.set_xlabel("No. of documents")
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
                parts = [p.strip() for p in entry.split(",")]
                # Prefer the 2nd token (university) over the 1st (department);
                # fall back to the 1st if there is only one token.
                inst = parts[1] if len(parts) >= 2 else parts[0] if parts else ""
                if inst:
                    institutions.append(inst)

        if institutions:
            inst_counts = pd.Series(institutions).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            inst_counts.plot.barh(ax=ax, color="#4c72b0")
            ax.set_xlabel("No. of documents")
            ax.set_title("Top 20 Affiliations")
            ax.invert_yaxis()
            plt.tight_layout()
            fig.savefig(out / "top_affiliations.png", dpi=150)
            plt.close(fig)
