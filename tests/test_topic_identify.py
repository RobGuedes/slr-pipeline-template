"""Tests for pipeline.topic_identify — topic labeling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from pipeline.topic_identify import (
    get_topic_terms,
    generate_topic_label,
    get_all_topic_labels,
    export_topic_terms,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def mock_lda_model():
    """Mock a Gensim LdaModel."""
    model = MagicMock()
    # Mock show_topic to return list of (word, prob)
    # Topic 0: machine learning
    # Topic 1: deep learning
    def side_effect(topic_id, topn=10):
        if topic_id == 0:
            return [("machine", 0.5), ("learning", 0.4)]
        else:
            return [("deep", 0.6), ("learning", 0.3)]
    
    model.show_topic.side_effect = side_effect
    model.num_topics = 2
    return model


# ── get_topic_terms ───────────────────────────────────────────────────


class TestGetTopicTerms:
    def test_returns_list_of_terms(self):
        model = mock_lda_model()
        terms = get_topic_terms(model, 0)
        assert len(terms) == 2
        assert terms[0] == ("machine", 0.5)

    def test_calls_show_topic(self):
        model = mock_lda_model()
        get_topic_terms(model, 1, topn=5)
        model.show_topic.assert_called_with(1, topn=5)


# ── generate_topic_label ──────────────────────────────────────────────


class TestGenerateTopicLabel:
    def test_stub_returns_joined_top_terms(self):
        """For now, the stub just joins the top 3 words."""
        terms = [("machine", 0.5), ("learning", 0.4), ("algo", 0.1)]
        label = generate_topic_label(terms)
        # Stub logic: "machine, learning, algo" (or similar)
        # We'll assert it contains the words.
        assert "machine" in label
        assert "learning" in label


# ── get_all_topic_labels ──────────────────────────────────────────────


class TestGetAllTopicLabels:
    def test_iterates_all_topics(self):
        model = mock_lda_model()
        labels = get_all_topic_labels(model)
        assert len(labels) == 2
        assert 0 in labels
        assert 1 in labels
        # Check stub outcome
        assert "machine" in labels[0]
        assert "deep" in labels[1]


# ── export_topic_terms ───────────────────────────────────────────────


class TestExportTopicTerms:
    def test_returns_dataframe_with_expected_columns(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            df = export_topic_terms(model, Path(tmpdir))
        assert list(df.columns) == ["topic_id", "label", "rank", "term", "weight"]

    def test_dataframe_has_rows_for_all_topics_and_terms(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            df = export_topic_terms(model, Path(tmpdir))
        # 2 topics x 2 terms each = 4 rows
        assert len(df) == 4
        assert set(df["topic_id"]) == {0, 1}

    def test_ranks_are_one_indexed(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            df = export_topic_terms(model, Path(tmpdir))
        topic0 = df[df["topic_id"] == 0]
        assert list(topic0["rank"]) == [1, 2]

    def test_writes_csv_file(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_topic_terms(model, Path(tmpdir))
            csv_path = Path(tmpdir) / "topic_terms.csv"
            assert csv_path.exists()
            loaded = pd.read_csv(csv_path)
            assert len(loaded) == 4
            assert "weight" in loaded.columns

    def test_writes_json_file(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_topic_terms(model, Path(tmpdir))
            json_path = Path(tmpdir) / "topic_terms.json"
            assert json_path.exists()
            with open(json_path) as f:
                data = json.load(f)
            assert len(data) == 2  # 2 topics
            assert data[0]["topic_id"] == 0
            assert len(data[0]["terms"]) == 2
            assert "weight" in data[0]["terms"][0]

    def test_respects_topn_parameter(self):
        model = mock_lda_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            df = export_topic_terms(model, Path(tmpdir), topn=1)
        # Mock returns 2 terms but topn=1 passed to show_topic
        model.show_topic.assert_any_call(0, topn=1)

    def test_handles_numpy_float32_weights(self):
        """Regression test: gensim returns numpy float32, which json can't serialize."""
        model = MagicMock()
        # Return numpy types like real gensim does
        model.show_topic.return_value = [
            ("term1", np.float32(0.123)),
            ("term2", np.float32(0.456)),
        ]
        model.num_topics = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            export_topic_terms(model, Path(tmpdir))
            # Should not raise JSON serialization error
            json_path = Path(tmpdir) / "topic_terms.json"
            assert json_path.exists()
            with open(json_path) as f:
                data = json.load(f)
            # Verify weights are native Python floats (JSON serialization succeeded)
            assert isinstance(data[0]["terms"][0]["weight"], float)
            # Float32 -> Python float has precision differences
            assert abs(data[0]["terms"][0]["weight"] - 0.123) < 0.001
