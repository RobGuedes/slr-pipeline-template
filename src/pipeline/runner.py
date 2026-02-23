"""Step 9 — Orchestrator: run the full SLR pipeline.

Connects all modules:
1. Ingest (from raw data)
2. Preprocess (NLP)
3. Topic Modeling (LDA Sweep & Final Training)
4. Topic Identification (Labeling)
5. Topic Selection (Filtering)
6. Synthesis (Reporting & Visualization)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.ingest import ingest_all
from pipeline.metrics import PipelineMetrics
from pipeline.preprocess import setup_nltk, clean_text, create_corpus
from pipeline.topic_model import (
    perform_lda_sweep,
    train_final_model,
    select_top_candidates,
)
from pipeline.topic_identify import get_all_topic_labels, export_topic_terms
from pipeline.topic_select import assign_dominant_topic, filter_documents
from pipeline.quality_review import export_for_review
from pipeline.synthesis import (
    export_report_tex,
    export_report_json,
    plot_topics,
    plot_topic_audit,
    plot_bibliometrics,
)

if TYPE_CHECKING:
    pass

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig | None = None) -> None:
    """Execute the end-to-end SLR pipeline.

    - Loads raw data
    - Preprocesses text
    - Sweeps for optimal K (topics)
    - Trains final LDA model
    - Assigns topics and filters documents
    - Exports for manual review
    - Generates synthesis reports and plots
    """
    if config is None:
        config = PipelineConfig()

    logger.info("Starting pipeline run...")

    # ── Step 1 & 2: Ingest ─────────────────────────────────────────────
    logger.info(f"Ingesting raw data from {config.raw_dir}")
    df_raw, ingest_counts = ingest_all(config.raw_dir, config, return_counts=True)
    logger.info(f"Ingested {len(df_raw)} unique documents.")

    # ── Step 3: Preprocess ─────────────────────────────────────────────
    logger.info("Setting up NLTK resources...")
    setup_nltk()

    logger.info("Cleaning and tokenizing text...")
    # Apply cleaning to 'abstract' + 'title' combined?
    # Legacy likely used abstract only or both.
    # Let's combine for richer topic modeling.
    text_data = (df_raw["title"] + " " + df_raw["abstract"]).tolist()
    extra_stopwords = set(config.academic_stopwords) | set(config.domain_stopwords)
    tokens = [
        clean_text(doc, extra_stopwords=extra_stopwords, nouns_only=config.nouns_only)
        for doc in text_data
    ]

    logger.info("Creating dictionary and corpus...")
    dictionary, corpus = create_corpus(tokens)
    logger.info(f"Dictionary size: {len(dictionary)} unique tokens.")

    # ── Step 4: Topic Modeling (Sweep) ─────────────────────────────────
    logger.info(f"Performing LDA sweep for K={config.topic_range}...")
    sweep_results = perform_lda_sweep(
        corpus=corpus,
        id2word=dictionary,
        texts=tokens,
        k_values=list(range(*config.topic_range)),
        passes=config.num_passes_sweep,
        random_state=config.random_state,
    )

    # Generate topic audit plot (perplexity + coherence vs K)
    audit_plot_path = config.processed_dir / "topic_audit_plot.png"
    logger.info(f"Generating topic audit plot at {audit_plot_path}...")
    plot_topic_audit(sweep_results, audit_plot_path)

    # Identify top candidate Ks
    candidates = select_top_candidates(sweep_results, n=3)
    valid_ks = {r.k for r in sweep_results}
    best_k = candidates[0].k

    # Present candidates to user
    print("\n══════════════════════════════════════════════════")
    print("Top K candidates (ranked by coherence):\n")
    print(f"  {'Rank':>4} │ {'K':>3} │ {'Coherence':>9} │ {'Log Perplexity':>14}")
    print(f"  {'─' * 4}─┼─{'─' * 3}─┼─{'─' * 9}─┼─{'─' * 14}")
    for i, c in enumerate(candidates, 1):
        print(f"  {i:>4} │ {c.k:>3} │ {c.coherence:>9.4f} │ {c.perplexity:>14.3f}")
    print(f"\n  Audit plot saved to: {audit_plot_path}")
    print("══════════════════════════════════════════════════\n")

    # Prompt user for K selection
    attempts = 0
    optimal_k = best_k
    while attempts < 3:
        raw = input(f"Select K [default={best_k}]: ").strip()
        if raw == "":
            optimal_k = best_k
            break
        try:
            chosen = int(raw)
            if chosen in valid_ks:
                optimal_k = chosen
                break
            else:
                print(
                    f"  Invalid: K={chosen} not in sweep range. Valid: {sorted(valid_ks)}"
                )
        except ValueError:
            print("  Invalid input. Enter a number.")
        attempts += 1
    else:
        logger.warning(f"Max attempts reached. Using best K={best_k}.")
        optimal_k = best_k

    selected_coherence = next(r.coherence for r in sweep_results if r.k == optimal_k)
    logger.info(f"Selected K={optimal_k} (Coherence: {selected_coherence:.4f})")

    # ── Step 5: Final Model ────────────────────────────────────────────
    logger.info(f"Training final model with K={optimal_k}...")
    final_model = train_final_model(
        corpus=corpus,
        id2word=dictionary,
        num_topics=optimal_k,
        passes=config.num_passes_final,
        random_state=config.random_state,
    )

    # ── Step 6: Identify & Label ───────────────────────────────────────
    logger.info("Generating topic labels...")
    labels = get_all_topic_labels(final_model)
    for tid, label in labels.items():
        logger.info(f"Topic {tid}: {label}")

    logger.info("Exporting topic terms with weights...")
    export_topic_terms(final_model, config.processed_dir)

    # ── Step 7: Selection ──────────────────────────────────────────────
    logger.info("Assigning dominant topics and filtering...")
    df_topics = assign_dominant_topic(df_raw, final_model, corpus)

    # Map topic IDs to labels for readability?
    # Usually we keep ID for processing, map for display.

    df_selected, filter_stats = filter_documents(
        df_topics, config, full_df=df_topics, return_stats=True
    )
    logger.info(f"Selected {len(df_selected)} documents after filtering.")

    # Count papers per topic from final selected papers (after all filtering)
    papers_per_topic = df_selected["Dominant_Topic"].value_counts().to_dict()

    # ── Step 8: Quality Review Export ──────────────────────────────────
    review_path = config.processed_dir / "to_review.csv"
    logger.info(f"Exporting for manual review to {review_path}...")
    export_for_review(df_selected, review_path)

    # ── Step 9: Synthesis ──────────────────────────────────────────────
    logger.info("Generating synthesis reports...")

    # Build PipelineMetrics
    metrics = PipelineMetrics(
        scopus_raw=ingest_counts["scopus"],
        wos_raw=ingest_counts["wos"],
        duplicates_removed=ingest_counts["duplicates_removed"],
        unique_papers=ingest_counts["unique"],
        selected_k=optimal_k,
        coherence_score=selected_coherence,
        papers_per_topic=papers_per_topic,
        failed_topic_assignment=filter_stats["failed_topic_assignment"],
        passed_probability=filter_stats["passed_probability"],
        passed_citations=filter_stats["passed_citations"],
        papers_selected_strict=filter_stats["selected_strict"],
        papers_recovered=filter_stats["recovered"],
        papers_final=len(df_selected),
        year_min=int(df_selected["year"].min()) if len(df_selected) > 0 and pd.notna(df_selected["year"].min()) else 0,
        year_max=int(df_selected["year"].max()) if len(df_selected) > 0 and pd.notna(df_selected["year"].max()) else 0,
        total_citations=int(df_selected["cited_by"].sum()) if len(df_selected) > 0 else 0,
    )

    # Export both synthesis reports
    tex_path = config.processed_dir / "synthesis_report.tex"
    logger.info(f"Exporting synthesis report to {tex_path}...")
    export_report_tex(metrics, tex_path)

    json_path = config.processed_dir / "synthesis_metadata.json"
    logger.info(f"Exporting synthesis metadata to {json_path}...")
    export_report_json(metrics, json_path)

    # 2. Bibliometric plots
    logger.info("Generating bibliometric plots...")
    plot_bibliometrics(df_selected, config.processed_dir)

    # 3. Topic Plot
    plot_path = config.processed_dir / "topic_plot.html"
    logger.info(f"Generating topic map at {plot_path}...")
    plot_topics(final_model, corpus, dictionary, plot_path)

    logger.info("Pipeline run complete.")
