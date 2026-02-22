"""Step 9 — Synthesis: visualized analysis and reporting.

Uses pyLDAvis for interactive topic maps and custom matplotlib helpers for
bibliometric plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models

if TYPE_CHECKING:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel


def _plot_horizontal_bar(
    data: pd.Series,
    xlabel: str,
    title: str,
    output_path: Path | str,
) -> None:
    """Render a horizontal bar chart with unified project styling.

    Parameters
    ----------
    data : pd.Series
        Index = labels, values = counts. Assumed already sorted/truncated.
    xlabel : str
        Label for the x-axis.
    title : str
        Chart title.
    output_path : Path | str
        Where to save the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


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
            "total_citations": 0,
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
    output_path: Path | str,
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

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    k_values = [r.k for r in sweep_results]
    perplexities = [r.perplexity for r in sweep_results]
    coherences = [r.coherence for r in sweep_results]

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
    output_dir: Path | str,
) -> None:
    """Generate bibliometric plots and save them to output_dir.

    All plots use a unified visual identity. Horizontal bar charts use
    ``_plot_horizontal_bar``; the publication years chart uses vertical bars.

    Plots generated:
    - publication_years.png  (vertical bars)
    - top_authors.png        (horizontal bars)
    - top_sources.png        (horizontal bars)
    - top_countries.png      (horizontal bars)
    - top_affiliations.png   (horizontal bars)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Publication Years (vertical bars) ────────────────────────
    if "year" in df.columns and not df["year"].dropna().empty:
        year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        year_counts.plot.bar(ax=ax, color="#4c72b0")
        ax.set_xlabel("Year")
        ax.set_ylabel("No. of documents")
        ax.set_title("Publication Trends")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(out / "publication_years.png", dpi=150)
        plt.close(fig)

    # ── 2. Top Authors (horizontal bars) ────────────────────────────
    if "author" in df.columns:
        authors: list[str] = []
        for raw in df["author"].dropna():
            for name in str(raw).split(";"):
                name = name.strip()
                if name:
                    authors.append(name)
        if authors:
            author_counts = pd.Series(authors).value_counts().head(15)
            _plot_horizontal_bar(
                author_counts,
                "No. of documents",
                "Top 15 Authors",
                out / "top_authors.png",
            )

    # ── 3. Top Publication Sources (horizontal bars) ────────────────
    if "source_title" in df.columns:
        source_counts = df["source_title"].dropna().value_counts().head(15)
        if not source_counts.empty:
            _plot_horizontal_bar(
                source_counts,
                "No. of documents",
                "Top 15 Publication Sources",
                out / "top_sources.png",
            )

    # ── 4. Top Countries (horizontal bars) ──────────────────────────
    aff_col = "Affiliations" if "Affiliations" in df.columns else "affiliations"
    if aff_col in df.columns:
        countries: list[str] = []
        import re
        for raw in df[aff_col].dropna():
            cleaned = re.sub(r'\[.*?\]', '', str(raw))
            for entry in cleaned.split(";"):
                parts = [p.strip() for p in entry.split(",") if p.strip()]
                if parts:
                    country_raw = parts[-1]
                    c_up = country_raw.upper()
                    if "USA" in c_up or "UNITED STATES" in c_up:
                        country = "USA"
                    elif "CHINA" in c_up:
                        country = "China"
                    elif "UK" in c_up or "ENGLAND" in c_up:
                        country = "UK"
                    else:
                        # Try to remove numbers (e.g., zip codes)
                        words = [w for w in country_raw.split() if not any(c.isdigit() for c in w)]
                        country = " ".join(words)
                    
                    if country:
                        countries.append(country)
        if countries:
            country_counts = pd.Series(countries).value_counts().head(15)
            _plot_horizontal_bar(
                country_counts,
                "No. of documents",
                "Top 15 Countries",
                out / "top_countries.png",
            )

    # ── 5. Top Affiliations (horizontal bars) ───────────────────────
    if aff_col in df.columns:
        institutions: list[str] = []
        has_source = "source_db" in df.columns
        iterator = df[["source_db", aff_col]].dropna(subset=[aff_col]).itertuples() if has_source else df[[aff_col]].dropna().itertuples()
        
        for row in iterator:
            raw = getattr(row, aff_col)
            source = getattr(row, "source_db", "Unknown")
            cleaned = re.sub(r'\[.*?\]', '', str(raw))
            for entry in cleaned.split(";"):
                parts = [p.strip() for p in entry.split(",") if p.strip()]
                if parts:
                    if source == "Scopus":
                        inst = parts[1] if len(parts) >= 2 else parts[0]
                    else:
                        inst = parts[0]
                    
                    if inst:
                        institutions.append(inst)
        if institutions:
            inst_counts = pd.Series(institutions).value_counts().head(15)
            _plot_horizontal_bar(
                inst_counts,
                "No. of documents",
                "Top 15 Affiliations",
                out / "top_affiliations.png",
            )
