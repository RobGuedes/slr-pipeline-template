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

from pipeline.config import PipelineConfig
from pipeline.ingest import ingest_all
from pipeline.preprocess import setup_nltk, clean_text, create_corpus
from pipeline.topic_model import perform_lda_sweep, train_final_model, select_top_candidates
from pipeline.topic_identify import get_all_topic_labels
from pipeline.topic_select import assign_dominant_topic, filter_documents
from pipeline.quality_review import export_for_review
from pipeline.synthesis import (
    generate_report,
    export_report_tex,
    plot_topics,
    plot_topic_audit,
    plot_bibliometrics,
)

if TYPE_CHECKING:
    from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    df_raw = ingest_all(config.raw_dir, config)
    logger.info(f"Ingested {len(df_raw)} unique documents.")
    
    # ── Step 3: Preprocess ─────────────────────────────────────────────
    logger.info("Setting up NLTK resources...")
    setup_nltk()
    
    logger.info("Cleaning and tokenizing text...")
    # Apply cleaning to 'abstract' + 'title' combined?
    # Legacy likely used abstract only or both.
    # Let's combine for richer topic modeling.
    text_data = (df_raw["title"] + " " + df_raw["abstract"]).tolist()
    tokens = [clean_text(doc) for doc in text_data]
    
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
                print(f"  Invalid: K={chosen} not in sweep range. Valid: {sorted(valid_ks)}")
        except ValueError:
            print(f"  Invalid input. Enter a number.")
        attempts += 1
    else:
        logger.warning(f"Max attempts reached. Using best K={best_k}.")
        optimal_k = best_k

    selected_coherence = next(
        r.coherence for r in sweep_results if r.k == optimal_k
    )
    logger.info(f"Selected K={optimal_k} (Coherence: {selected_coherence:.4f})")
    
    # ── Step 5: Final Model ────────────────────────────────────────────
    logger.info(f"Training final model with K={optimal_k}...")
    final_model = train_final_model(
        corpus=corpus,
        id2word=dictionary,
        num_topics=optimal_k,
        passes=config.num_passes_final,
        random_state=config.random_state
    )
    
    # ── Step 6: Identify & Label ───────────────────────────────────────
    logger.info("Generating topic labels...")
    labels = get_all_topic_labels(final_model)
    for tid, label in labels.items():
        logger.info(f"Topic {tid}: {label}")
        
    # ── Step 7: Selection ──────────────────────────────────────────────
    logger.info("Assigning dominant topics and filtering...")
    df_topics = assign_dominant_topic(df_raw, final_model, corpus)
    
    # Map topic IDs to labels for readability? 
    # Usually we keep ID for processing, map for display.
    
    df_selected = filter_documents(df_topics, config, full_df=df_topics)
    logger.info(f"Selected {len(df_selected)} documents after filtering.")
    
    # ── Step 8: Quality Review Export ──────────────────────────────────
    review_path = config.processed_dir / "to_review.csv"
    logger.info(f"Exporting for manual review to {review_path}...")
    export_for_review(df_selected, review_path)
    
    # ── Step 9: Synthesis ──────────────────────────────────────────────
    logger.info("Generating synthesis reports...")
    
    # 1. Report stats
    stats = generate_report(df_selected)
    logger.info(f"Report Stats: {stats}")
    
    # Export for LaTeX
    tex_path = config.processed_dir / "synthesis_report.tex"
    logger.info(f"Exporting synthesis report to {tex_path}...")
    export_report_tex(stats, tex_path)
    
    # 2. Bibliometric plots
    logger.info("Generating bibliometric plots...")
    plot_bibliometrics(df_selected, config.processed_dir)
        
    # 3. Topic Plot
    plot_path = config.processed_dir / "topic_plot.html"
    logger.info(f"Generating topic map at {plot_path}...")
    plot_topics(final_model, corpus, dictionary, plot_path)
    
    logger.info("Pipeline run complete.")
