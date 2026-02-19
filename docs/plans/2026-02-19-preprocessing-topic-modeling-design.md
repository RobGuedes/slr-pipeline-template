# Preprocessing & Topic Modeling Fixes — Design

**Date:** 2026-02-19
**Context:** Audit of `preprocess.py` and `topic_model.py` against the legacy notebook
`01. Topic Modelling_20241028 copy.ipynb` revealed 3 significant gaps and 1 confirmation.

## Decisions

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | Tokenizer | `RegexpTokenizer(r'\w+')` | Keeps digits-in-words (lstm2, t5, arima) relevant to ML/finance domain |
| 2 | Lemmatization | POS-aware via `nltk.pos_tag()` + `get_wordnet_pos()` | Legacy approach. Correctly lemmatizes verbs/adjectives, not just nouns |
| 3 | Token dedup | `OrderedDict.fromkeys()` per document | SLR cares about concept presence, not repetition frequency |
| 4 | TF-IDF | Skip (not used for LDA) | Legacy uses TF-IDF only for SVD/PCA exploration, not as LDA input |
| 5 | Coherence | `c_v` with `texts=texts` | Legacy uses `c_v`. Produces positive 0-1 scores. More interpretable than `u_mass` |
| 6 | HTML/number removal | Skip | Data is clean bibliographic exports. Number removal conflicts with tokenizer keeping digits |

## What Changes

### `preprocess.py`

**`setup_nltk()`** — add `averaged_perceptron_tagger_eng` to auto-download list.

**`clean_text()`** — rewrite pipeline:
1. `unidecode(text).lower()` (unchanged)
2. `RegexpTokenizer(r'\w+').tokenize(text)` (was: `re.findall`)
3. Stopword removal (unchanged)
4. POS tagging: `nltk.pos_tag(tokens)` + `get_wordnet_pos(tag)`
5. Lemmatize with POS: `lemmatizer.lemmatize(token, pos)` (was: noun-only)
6. Deduplicate: `list(OrderedDict.fromkeys(tokens))` (new)
7. Min-length filter `len(t) > 2` (unchanged)

### `topic_model.py`

**`perform_lda_sweep()`** — new `texts` parameter, switch to `c_v`:
```python
def perform_lda_sweep(
    corpus, id2word, k_values,
    texts: list[list[str]],  # NEW — required for c_v
    ...
) -> list[SweepResult]:
    ...
    coherence_model = CoherenceModel(
        model=model, texts=texts, dictionary=id2word, coherence="c_v"
    )
```

### `runner.py`

Pass `tokens` to `perform_lda_sweep()`:
```python
sweep_results = perform_lda_sweep(
    corpus=corpus, id2word=dictionary, k_values=...,
    texts=tokens,  # NEW
    ...
)
```

## What Does NOT Change

- `create_corpus()` — BoW via Gensim Dictionary (correct, matches legacy)
- `config.py` — no new fields needed for these fixes
- `synthesis.py` — `plot_topic_audit()` Y-axis "Log Perplexity" is already correct
- `topic_select.py`, `quality_review.py` — untouched
