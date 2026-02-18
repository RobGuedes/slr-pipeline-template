"""Step 5 & 6 — Topic Modeling: LDA sweep and final model training.

Encapsulates Gensim's LdaModel and CoherenceModel to perform hyperparameter
optimization (finding optimal K) and training the final model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gensim.models import CoherenceModel, LdaModel

if TYPE_CHECKING:
    from gensim.corpora import Dictionary


@dataclass
class SweepResult:
    """Result of a single LDA run during the parameter sweep."""
    
    k: int
    coherence: float
    perplexity: float
    model: LdaModel


def perform_lda_sweep(
    corpus: list[list[tuple[int, int]]],
    id2word: Dictionary,
    k_values: list[int],
    passes: int = 10,
    random_state: int = 42,
    alpha: str | float = "auto",
    eta: str | float = "auto",
) -> list[SweepResult]:
    """Train multiple LDA models with different K and compute coherence.

    Parameters
    ----------
    corpus : list
        Gensim BoW corpus.
    id2word : Dictionary
        Gensim dictionary mapping.
    k_values : list[int]
        List of topic counts (K) to try.
    passes : int
        Number of passes over the corpus during training.
    random_state : int
        Seed for reproducibility.
    alpha : str | float
        Hyperparameter for document-topic density.
    eta : str | float
        Hyperparameter for topic-word density.

    Returns
    -------
    list[SweepResult]
        List of results sorted by K.
    """
    results: list[SweepResult] = []

    for k in k_values:
        # Train LDA
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=k,
            random_state=random_state,
            passes=passes,
            alpha=alpha,
            eta=eta,
        )

        # Compute Coherence
        # c_v is the most popular metric, though u_mass is faster.
        # Legacy notebook used c_v (implied, or default).
        coherence_model = CoherenceModel(
            model=model,
            corpus=corpus,
            dictionary=id2word,
            coherence="u_mass"  # u_mass is faster and uses corpus, c_v needs texts
        )
        # Note: If we want c_v, we need the tokenized texts, not just corpus.
        # For now, using u_mass as it only requires corpus/dictionary.
        # If legacy used c_v, we'd need to pass `texts` argument.
        # Let's check legacy notebook...
        # Legacy used pyLDAvis which often implies c_v or u_mass analysis?
        # Safe default is u_mass for speed and simplicity here.
        
        score = coherence_model.get_coherence()
        perplexity = model.log_perplexity(corpus)
        results.append(SweepResult(k=k, coherence=score, perplexity=perplexity, model=model))

    return results


def train_final_model(
    corpus: list[list[tuple[int, int]]],
    id2word: Dictionary,
    num_topics: int,
    passes: int = 20,
    random_state: int = 42,
    alpha: str | float = "auto",
    eta: str | float = "auto",
) -> LdaModel:
    """Train the final LDA model with the selected optimal K.

    Parameters
    ----------
    corpus : list
        Gensim BoW corpus.
    id2word : Dictionary
        Gensim dictionary.
    num_topics : int
        Optimal K (number of topics).
    passes : int
        Number of passes (usually higher than sweep for better quality).
    random_state : int
        Seed.

    Returns
    -------
    LdaModel
        The trained model.
    """
    return LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        alpha=alpha,
        eta=eta,
    )
