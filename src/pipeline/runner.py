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
from pipeline.topic_model import perform_lda_sweep, train_final_model
from pipeline.topic_identify import get_all_topic_labels
from pipeline.topic_select import assign_dominant_topic, filter_documents
from pipeline.quality_review import export_for_review
from pipeline.synthesis import (
    convert_to_litstudy,
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
        k_values=list(range(*config.topic_range)),
        passes=config.num_passes_sweep,
        random_state=config.random_state
    )
    
    # Generate topic audit plot (perplexity + coherence vs K)
    audit_plot_path = config.processed_dir / "topic_audit_plot.png"
    logger.info(f"Generating topic audit plot at {audit_plot_path}...")
    plot_topic_audit(sweep_results, audit_plot_path)

    # Simple logic to pick best K: max coherence
    best_result = max(sweep_results, key=lambda x: x.coherence)
    optimal_k = best_result.k
    logger.info(f"Optimal K found: {optimal_k} (Coherence: {best_result.coherence:.4f})")
    
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
    
    df_selected = filter_documents(df_topics, config)
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
    
    # 2. LitStudy conversion (demo)
    try:
        docs_lit = convert_to_litstudy(df_selected)
        logger.info(f"Converted to LitStudy DocumentSet with {len(docs_lit)} docs.")
        
        # Generate bibliometric plots
        logger.info("Generating bibliometric plots (trends, authors, affiliations)...")
        plot_bibliometrics(df_selected, docs_lit, config.processed_dir)
        
    except ImportError:
        logger.warning("LitStudy not installed, skipping advanced bibliometrics.")
        
    # 3. Topic Plot
    plot_path = config.processed_dir / "topic_plot.html"
    logger.info(f"Generating topic map at {plot_path}...")
    plot_topics(final_model, corpus, dictionary, plot_path)
    
    logger.info("Pipeline run complete.")
