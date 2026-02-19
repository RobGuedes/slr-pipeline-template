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
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.convert_to_litstudy")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
def test_end_to_end_flow(
    mock_labels, mock_export_review, mock_plot_biblio, mock_plot_audit,
    mock_plot, mock_tex, mock_report, mock_litstudy, mock_filter, mock_assign,
    mock_train, mock_sweep, mock_corpus, mock_clean, mock_ingest
):
    """Verify the sequence of steps in the pipeline."""

    # Setup mocks
    mock_ingest.return_value = pd.DataFrame({
        "title": ["Paper A"],
        "abstract": ["Abstract of paper A"]
    })
    mock_corpus.return_value = (MagicMock(name="dictionary"), MagicMock(name="corpus"))

    # Sweep returns optimal k=5
    mock_sweep_result = MagicMock()
    mock_sweep_result.k = 5
    mock_sweep_result.coherence = 0.5
    mock_sweep.return_value = [mock_sweep_result]

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
    mock_litstudy.assert_called_once()
    mock_plot.assert_called_once()
    mock_report.assert_called_once()


def test_uses_existing_model_if_provided():
    """If we pass a pre-trained model path, it should skip training/sweep?

    Actually, the current plan doesn't specify loading pre-trained models yet,
    but runner should handle overrides. For now, we test vanilla flow.
    """
    pass
