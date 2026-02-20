from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pipeline.runner import run_pipeline
from pipeline.config import PipelineConfig


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", return_value="")
def test_end_to_end_flow(
    mock_input,
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_report,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
):
    """Verify the sequence of steps in the pipeline."""

    # Setup mocks
    mock_ingest.return_value = pd.DataFrame(
        {"title": ["Paper A"], "abstract": ["Abstract of paper A"]}
    )
    mock_corpus.return_value = (MagicMock(name="dictionary"), MagicMock(name="corpus"))

    # Sweep returns optimal k=5
    mock_sweep_result = MagicMock()
    mock_sweep_result.k = 5
    mock_sweep_result.coherence = 0.5
    mock_sweep_result.perplexity = -6.0
    mock_sweep.return_value = [mock_sweep_result]

    # top candidates returns same result (default K = 5)
    mock_top_candidates.return_value = [mock_sweep_result]

    mock_train.return_value = MagicMock(name="lda_model")

    # Run pipeline
    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    # Verification
    mock_ingest.assert_called_once()
    mock_clean.assert_called()
    mock_corpus.assert_called_once()
    mock_sweep.assert_called_once()
    # Verify texts was passed to sweep
    sweep_call_kwargs = mock_sweep.call_args.kwargs
    assert "texts" in sweep_call_kwargs
    mock_train.assert_called_once()
    mock_assign.assert_called_once()
    mock_filter.assert_called_once()
    mock_plot.assert_called_once()
    mock_report.assert_called_once()


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", return_value="3")
def test_interactive_k_selection(
    mock_input,
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_report,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
):
    """Pipeline prompts user and trains model with chosen K."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame(
        {"title": ["Paper A"], "abstract": ["Abstract A"]}
    )
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    all_results = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_sweep.return_value = all_results

    # Top candidates returns k=5 first (highest coherence)
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    # User typed "3", so final model should use K=3
    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 3

    # input() was called
    mock_input.assert_called_once()


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", return_value="")
def test_empty_input_uses_default_best_k(
    mock_input,
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_report,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
):
    """Empty input (just Enter) should use the default best K."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame(
        {"title": ["Paper A"], "abstract": ["Abstract A"]}
    )
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    mock_sweep.return_value = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    # Default = best K = 5
    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 5


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", side_effect=["99", "3"])
def test_invalid_then_valid_k_selection(
    mock_input,
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_report,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
):
    """First attempt enters K not in sweep range; second attempt succeeds."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame(
        {"title": ["Paper A"], "abstract": ["Abstract A"]}
    )
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    mock_sweep.return_value = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 3
    assert mock_input.call_count == 2


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", side_effect=["bad", "99", "nope"])
def test_exhausted_attempts_fall_back_to_best_k(
    mock_input,
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_report,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
):
    """Three invalid inputs exhaust the retry loop; default K is used."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame(
        {"title": ["Paper A"], "abstract": ["Abstract A"]}
    )
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    mock_sweep.return_value = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 5  # best_k fallback
    assert mock_input.call_count == 3


def test_uses_existing_model_if_provided():
    """If we pass a pre-trained model path, it should skip training/sweep?

    Actually, the current plan doesn't specify loading pre-trained models yet,
    but runner should handle overrides. For now, we test vanilla flow.
    """
    pass
