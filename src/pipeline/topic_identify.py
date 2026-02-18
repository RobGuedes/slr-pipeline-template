"""Step 5 (part 2) — Topic Identification: naming the topics.

Extracts top terms from the LDA model and generates labels (names) for them.
Currently uses a stub (top 3 terms) as a placeholder for LLM-based labeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gensim.models import LdaModel


def get_topic_terms(
    model: LdaModel,
    topic_id: int,
    topn: int = 10
) -> list[tuple[str, float]]:
    """Get the top *N* terms and their probabilities for a topic.

    Parameters
    ----------
    model : LdaModel
        The trained Gensim LDA model.
    topic_id : int
        The ID of the topic (0 to K-1).
    topn : int
        Number of top terms to retrieve.

    Returns
    -------
    list[tuple[str, float]]
        List of (word, probability) pairs.
    """
    return model.show_topic(topic_id, topn=topn)


def generate_topic_label(terms: list[tuple[str, float]]) -> str:
    """Generate a human-readable label for a topic based on its terms.

    Currently a stub that joins the top 3 terms.
    Future: Interface with an LLM (e.g. OpenAI) to generate a short phrase.

    Parameters
    ----------
    terms : list[tuple[str, float]]
        List of (word, probability) pairs.

    Returns
    -------
    str
        Generated label, e.g. "machine, learning, algorithm".
    """
    # Take top 3 terms by probability (terms are usually sorted by gensim)
    # Just in case, sort by prob descending
    sorted_terms = sorted(terms, key=lambda x: x[1], reverse=True)
    top_words = [word for word, prob in sorted_terms[:3]]
    return ", ".join(top_words)


def get_all_topic_labels(
    model: LdaModel,
    topn: int = 10
) -> dict[int, str]:
    """Generate labels for all topics in the model.

    Parameters
    ----------
    model : LdaModel
        Trained LDA model.
    topn : int
        Number of terms to consider for labeling.

    Returns
    -------
    dict[int, str]
        Mapping of {topic_id: label}.
    """
    labels = {}
    for topic_id in range(model.num_topics):
        terms = get_topic_terms(model, topic_id, topn=topn)
        labels[topic_id] = generate_topic_label(terms)
    return labels
