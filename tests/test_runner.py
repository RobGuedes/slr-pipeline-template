from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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
    df = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract of paper A"],
        "year": [2020],
        "cited_by": [10],
    })
    mock_ingest.return_value = (
        df,
        {"scopus": 1, "wos": 1, "duplicates_removed": 1, "unique": 1}
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

    # Mock assign to return df with Dominant_Topic
    df_topics = df.copy()
    df_topics["Dominant_Topic"] = [0]
    df_topics["Topic_Prob"] = [0.9]
    mock_assign.return_value = df_topics

    # Mock filter to return tuple
    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 1,
        "passed_citations": 1,
        "selected_strict": 1,
        "recovered": 0,
    }
    mock_filter.return_value = (df_topics, filter_stats)

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


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
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

    df = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract A"],
        "year": [2020],
        "cited_by": [10],
    })
    mock_ingest.return_value = (
        df,
        {"scopus": 1, "wos": 1, "duplicates_removed": 1, "unique": 1}
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

    # Mock assign and filter
    df_topics = df.copy()
    df_topics["Dominant_Topic"] = [0]
    df_topics["Topic_Prob"] = [0.9]
    mock_assign.return_value = df_topics

    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 1,
        "passed_citations": 1,
        "selected_strict": 1,
        "recovered": 0,
    }
    mock_filter.return_value = (df_topics, filter_stats)

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

    df = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract A"],
        "year": [2020],
        "cited_by": [10],
    })
    mock_ingest.return_value = (
        df,
        {"scopus": 1, "wos": 1, "duplicates_removed": 1, "unique": 1}
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

    # Mock assign and filter
    df_topics = df.copy()
    df_topics["Dominant_Topic"] = [0]
    df_topics["Topic_Prob"] = [0.9]
    mock_assign.return_value = df_topics

    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 1,
        "passed_citations": 1,
        "selected_strict": 1,
        "recovered": 0,
    }
    mock_filter.return_value = (df_topics, filter_stats)

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

    df = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract A"],
        "year": [2020],
        "cited_by": [10],
    })
    mock_ingest.return_value = (
        df,
        {"scopus": 1, "wos": 1, "duplicates_removed": 1, "unique": 1}
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

    # Mock assign and filter
    df_topics = df.copy()
    df_topics["Dominant_Topic"] = [0]
    df_topics["Topic_Prob"] = [0.9]
    mock_assign.return_value = df_topics

    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 1,
        "passed_citations": 1,
        "selected_strict": 1,
        "recovered": 0,
    }
    mock_filter.return_value = (df_topics, filter_stats)

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

    df = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract A"],
        "year": [2020],
        "cited_by": [10],
    })
    mock_ingest.return_value = (
        df,
        {"scopus": 1, "wos": 1, "duplicates_removed": 1, "unique": 1}
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

    # Mock assign and filter
    df_topics = df.copy()
    df_topics["Dominant_Topic"] = [0]
    df_topics["Topic_Prob"] = [0.9]
    mock_assign.return_value = df_topics

    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 1,
        "passed_citations": 1,
        "selected_strict": 1,
        "recovered": 0,
    }
    mock_filter.return_value = (df_topics, filter_stats)

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


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
def test_exports_synthesis_metadata_json(
    mock_labels,
    mock_export_review,
    mock_plot_biblio,
    mock_plot_audit,
    mock_plot,
    mock_tex,
    mock_filter,
    mock_assign,
    mock_train,
    mock_top_candidates,
    mock_sweep,
    mock_corpus,
    mock_clean,
    mock_ingest,
    tmp_path,
    monkeypatch,
):
    """Verify runner exports synthesis_metadata.json."""
    import json
    from pipeline.topic_model import SweepResult

    # Mock user input for K selection
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # Setup mocks
    mock_ingest.return_value = (
        pd.DataFrame({
            "title": ["Paper A", "Paper B"],
            "abstract": ["Abstract A", "Abstract B"],
            "year": [2020, 2021],
            "cited_by": [10, 20],
        }),
        {"scopus": 3, "wos": 2, "duplicates_removed": 3, "unique": 2},
    )
    mock_corpus.return_value = (MagicMock(name="dictionary"), MagicMock(name="corpus"))

    sweep_results = [SweepResult(k=5, coherence=0.5, perplexity=-6.0)]
    mock_sweep.return_value = sweep_results
    mock_top_candidates.return_value = sweep_results

    mock_train.return_value = MagicMock(name="lda_model")

    # Mock assign_dominant_topic to return a dataframe with topic assignments
    df_topics = pd.DataFrame({
        "title": ["Paper A", "Paper B"],
        "abstract": ["Abstract A", "Abstract B"],
        "year": [2020, 2021],
        "cited_by": [10, 20],
        "Dominant_Topic": [0, 1],
        "Topic_Prob": [0.9, 0.8],
    })
    mock_assign.return_value = df_topics

    # Mock filter_documents to return filtered df and stats
    df_selected = df_topics.copy()
    filter_stats = {
        "failed_topic_assignment": 0,
        "passed_probability": 2,
        "passed_citations": 2,
        "selected_strict": 2,
        "recovered": 0,
    }
    mock_filter.return_value = (df_selected, filter_stats)

    # Run pipeline
    config = PipelineConfig(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "proc")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    run_pipeline(config)

    # Verify JSON file was created with correct structure
    json_path = config.processed_dir / "synthesis_metadata.json"
    assert json_path.exists()

    with open(json_path) as f:
        data = json.load(f)

    assert "ingestion" in data
    assert "topic_modeling" in data
    assert "filtering" in data
    assert "final_dataset" in data
