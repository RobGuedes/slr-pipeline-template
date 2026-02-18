"""Tests for pipeline.topic_identify — topic labeling."""

from unittest.mock import MagicMock

from pipeline.topic_identify import get_topic_terms, generate_topic_label, get_all_topic_labels


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
