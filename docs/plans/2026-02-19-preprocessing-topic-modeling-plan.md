# Preprocessing & Topic Modeling Fixes — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `preprocess.py` and `topic_model.py` with the legacy notebook methodology — POS-aware lemmatization, RegexpTokenizer, per-document token dedup, and `c_v` coherence.

**Architecture:** Three independent changes: (1) rewrite `clean_text()` in `preprocess.py` with the correct NLP pipeline, (2) switch `perform_lda_sweep()` to `c_v` coherence with a new `texts` parameter, (3) update `runner.py` to pass tokenized texts to the sweep. Tests updated first (TDD), then implementation.

**Tech Stack:** Python 3.13, NLTK (RegexpTokenizer, pos_tag, WordNetLemmatizer), Gensim (CoherenceModel with c_v), pytest

---

### Task 1: Update `setup_nltk()` to download POS tagger

**Files:**
- Modify: `src/pipeline/preprocess.py:19-34`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add to `tests/test_preprocess.py` at the top level (outside the classes):

```python
def test_setup_nltk_downloads_pos_tagger():
    """setup_nltk should make POS tagger available."""
    setup_nltk()
    # If this doesn't raise LookupError, the resource is present
    import nltk
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_preprocess.py::test_setup_nltk_downloads_pos_tagger -v`
Expected: FAIL with `LookupError` (resource not found, since `setup_nltk` doesn't download it yet)

> Note: If the tagger is already installed on the system, this test will pass immediately. That's fine — it means the resource is available. The point is that `setup_nltk()` should guarantee it.

**Step 3: Add the POS tagger download to `setup_nltk()`**

In `src/pipeline/preprocess.py`, add after the `omw-1.4` block (after line 34):

```python
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_preprocess.py::test_setup_nltk_downloads_pos_tagger -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy
git add src/pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): add POS tagger to setup_nltk downloads"
```

---

### Task 2: Rewrite `clean_text()` — RegexpTokenizer + POS lemmatization + dedup

**Files:**
- Modify: `src/pipeline/preprocess.py:37-83`
- Modify: `tests/test_preprocess.py` (update `TestCleanText`)

**Step 1: Update existing tests and add new ones for the new behavior**

Replace the entire `TestCleanText` class in `tests/test_preprocess.py` with:

```python
class TestCleanText:
    @pytest.fixture(scope="class", autouse=True)
    def setup_nltk_data(self):
        """Ensure NLTK data is present for tests."""
        setup_nltk()

    def test_lowercases(self):
        tokens = clean_text("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_punctuation(self):
        # RegexpTokenizer(r'\w+') splits on non-word chars
        tokens = clean_text("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens

    def test_keeps_digits_in_words(self):
        # RegexpTokenizer keeps 'lstm2', 't5' etc.
        tokens = clean_text("lstm2 model works")
        assert "lstm2" in tokens

    def test_lemmatizes_verbs_with_pos(self):
        # POS-aware: "running" (verb) -> "run"
        tokens = clean_text("The cats are running quickly")
        assert "cat" in tokens
        assert "run" in tokens

    def test_lemmatizes_nouns(self):
        tokens = clean_text("studies about algorithms")
        assert "study" in tokens
        assert "algorithm" in tokens

    def test_removes_stopwords(self):
        tokens = clean_text("the cat sat on the mat")
        assert "the" not in tokens
        assert "cat" in tokens

    def test_removes_short_tokens(self):
        # Tokens with len <= 2 are removed after lemmatization
        tokens = clean_text("ab cde fg")
        assert "ab" not in tokens
        assert "fg" not in tokens
        assert "cde" in tokens

    def test_deduplicates_tokens_per_document(self):
        # Same word repeated should appear only once
        tokens = clean_text("model model model training model")
        assert tokens.count("model") == 1

    def test_dedup_preserves_order(self):
        # First occurrence order should be preserved
        tokens = clean_text("alpha beta alpha gamma beta")
        assert tokens.index("alpha") < tokens.index("beta")
        assert tokens.index("beta") < tokens.index("gamma")

    def test_handles_empty(self):
        assert clean_text("") == []

    def test_handles_none_like(self):
        assert clean_text("   ") == []

    def test_return_type_is_list(self):
        assert isinstance(clean_text("test something here"), list)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_preprocess.py::TestCleanText -v`
Expected: FAIL — `test_keeps_digits_in_words`, `test_lemmatizes_verbs_with_pos`, `test_deduplicates_tokens_per_document`, `test_dedup_preserves_order` will fail. Others may also fail due to changed behavior.

**Step 3: Rewrite `clean_text()` in `src/pipeline/preprocess.py`**

Replace the imports at the top (lines 9-16) with:

```python
import warnings
from collections import OrderedDict

import nltk
from gensim.corpora import Dictionary
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode
```

Remove the `import re` line (no longer needed).

Replace the entire `clean_text` function (lines 37-83) with:

```python
# Module-level tokenizer instance (stateless, safe to reuse)
_tokenizer = RegexpTokenizer(r"\w+")


def _get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tag to WordNet POS for lemmatization."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text: str) -> list[str]:
    """Clean, tokenize, POS-lemmatize, and deduplicate a single document.

    Pipeline (matches legacy notebook methodology):
    1. Lowercase & unidecode (remove accents)
    2. Tokenize with RegexpTokenizer(r'\\w+')
    3. Remove stopwords
    4. POS-tag and lemmatize with correct part-of-speech
    5. Remove duplicate tokens (preserve first-occurrence order)
    6. Remove short tokens (len <= 2)

    Parameters
    ----------
    text : str
        Raw text (e.g. abstract + title).

    Returns
    -------
    list[str]
        List of cleaned, unique tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. Normalize
    text = unidecode(text).lower()

    # 2. Tokenize (keeps digits-in-words like lstm2, t5)
    tokens = _tokenizer.tokenize(text)

    # 3. Stopwords
    stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stops]

    # 4. POS-aware lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)
    tokens = [
        lemmatizer.lemmatize(token, _get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]

    # 5. Deduplicate (preserve order)
    tokens = list(OrderedDict.fromkeys(tokens))

    # 6. Length filter
    tokens = [t for t in tokens if len(t) > 2]

    return tokens
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy
git add src/pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): POS-aware lemmatization, RegexpTokenizer, per-doc dedup

Aligns clean_text() with legacy notebook methodology:
- RegexpTokenizer(r'\w+') keeps digits-in-words (lstm2, t5)
- POS-tag + lemmatize (verbs, adjectives, not just nouns)
- OrderedDict.fromkeys() deduplicates tokens per document"
```

---

### Task 3: Switch `perform_lda_sweep()` to `c_v` coherence

**Files:**
- Modify: `src/pipeline/topic_model.py:28-95`
- Modify: `tests/test_topic_model.py`

**Step 1: Update existing tests to expect `texts` parameter and `c_v`**

In `tests/test_topic_model.py`, update fixtures and tests:

Add a `mock_texts` fixture after the existing fixtures:

```python
@pytest.fixture
def mock_texts():
    """Tokenized texts for c_v coherence."""
    return [["machine", "learning"], ["deep", "learning"], ["topic", "model"]]
```

Update `TestPerformLdaSweep` — both tests need to pass `texts` and verify `c_v` is used:

```python
class TestPerformLdaSweep:
    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_sweeps_range_of_k(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify it iterates over the requested range of K."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.return_value = -7.0
        MockLda.return_value = mock_lda_instance

        mock_coherence_instance = MagicMock()
        mock_coherence_instance.get_coherence.return_value = 0.5
        MockCoherence.return_value = mock_coherence_instance

        k_values = [2, 3]
        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=k_values,
            passes=1,
            random_state=42,
        )

        assert len(results) == 2
        assert results[0].k == 2
        assert results[1].k == 3
        assert MockLda.call_count == 2

        call_args_list = MockLda.call_args_list
        assert call_args_list[0].kwargs["num_topics"] == 2
        assert call_args_list[1].kwargs["num_topics"] == 3

    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_uses_c_v_coherence(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify CoherenceModel is called with coherence='c_v' and texts."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.return_value = -7.0
        MockLda.return_value = mock_lda_instance

        mock_coherence_instance = MagicMock()
        mock_coherence_instance.get_coherence.return_value = 0.5
        MockCoherence.return_value = mock_coherence_instance

        perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=[2],
            passes=1,
        )

        # Verify CoherenceModel was called with c_v and texts
        MockCoherence.assert_called_once()
        call_kwargs = MockCoherence.call_args.kwargs
        assert call_kwargs["coherence"] == "c_v"
        assert call_kwargs["texts"] == mock_texts

    @patch("pipeline.topic_model.LdaModel")
    @patch("pipeline.topic_model.CoherenceModel")
    def test_returns_coherence_and_perplexity(self, MockCoherence, MockLda, mock_corpus, mock_dictionary, mock_texts):
        """Verify it captures coherence score and perplexity."""
        mock_lda_instance = MagicMock()
        mock_lda_instance.log_perplexity.side_effect = [-7.0, -6.5]
        MockLda.return_value = mock_lda_instance

        mock_coherence = MockCoherence.return_value
        mock_coherence.get_coherence.side_effect = [0.4, 0.6]

        results = perform_lda_sweep(
            corpus=mock_corpus,
            id2word=mock_dictionary,
            texts=mock_texts,
            k_values=[2, 3],
            passes=1,
        )

        assert results[0].coherence == 0.4
        assert results[1].coherence == 0.6
        assert results[0].perplexity == -7.0
        assert results[1].perplexity == -6.5
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_topic_model.py::TestPerformLdaSweep -v`
Expected: FAIL — `perform_lda_sweep()` does not accept `texts` parameter

**Step 3: Update `perform_lda_sweep()` signature and coherence call**

In `src/pipeline/topic_model.py`, update `perform_lda_sweep`:

Change the signature (lines 28-36) to add `texts` parameter:

```python
def perform_lda_sweep(
    corpus: list[list[tuple[int, int]]],
    id2word: Dictionary,
    texts: list[list[str]],
    k_values: list[int],
    passes: int = 10,
    random_state: int = 42,
    alpha: str | float = "auto",
    eta: str | float = "auto",
) -> list[SweepResult]:
```

Update the docstring to include:
```
    texts : list[list[str]]
        Tokenized documents (required for c_v coherence).
```

Replace the CoherenceModel call (lines 78-83) with:

```python
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=id2word,
            coherence="c_v",
        )
```

Remove the old comment block (lines 84-89) about u_mass vs c_v since the decision is now made.

**Step 4: Run tests to verify they pass**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_topic_model.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy
git add src/pipeline/topic_model.py tests/test_topic_model.py
git commit -m "feat(topic_model): switch to c_v coherence, add texts parameter

CoherenceModel now uses coherence='c_v' with texts instead of
u_mass with corpus. c_v produces positive 0-1 scores and
correlates better with human topic interpretability."
```

---

### Task 4: Update `runner.py` to pass `tokens` to sweep

**Files:**
- Modify: `src/pipeline/runner.py:79-85`
- Modify: `tests/test_runner.py`

**Step 1: Update the runner test to verify `texts` is passed**

In `tests/test_runner.py`, update `test_end_to_end_flow` — the `mock_sweep` call should receive `texts`:

After the existing `mock_sweep.assert_called_once()` line (line 50), add:

```python
    # Verify texts was passed to sweep
    sweep_call_kwargs = mock_sweep.call_args.kwargs
    assert "texts" in sweep_call_kwargs
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_runner.py::test_end_to_end_flow -v`
Expected: FAIL — `texts` not in sweep call kwargs

**Step 3: Update `runner.py` to pass `tokens` to `perform_lda_sweep`**

In `src/pipeline/runner.py`, update the sweep call (lines 79-85):

```python
    sweep_results = perform_lda_sweep(
        corpus=corpus,
        id2word=dictionary,
        texts=tokens,
        k_values=list(range(*config.topic_range)),
        passes=config.num_passes_sweep,
        random_state=config.random_state,
    )
```

**Step 4: Run all tests to verify everything passes**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(runner): pass tokenized texts to LDA sweep for c_v coherence"
```

---

### Task 5: Remove `SweepResult.model` field (memory optimization)

**Files:**
- Modify: `src/pipeline/topic_model.py:18-25, 93`
- Modify: `tests/test_topic_model.py`

**Step 1: Write the failing test**

Add to `tests/test_topic_model.py`:

```python
class TestSweepResult:
    def test_sweep_result_stores_metrics_only(self):
        """SweepResult should NOT store the full LdaModel object."""
        from pipeline.topic_model import SweepResult
        result = SweepResult(k=5, coherence=0.42, perplexity=-7.1)
        assert result.k == 5
        assert result.coherence == 0.42
        assert result.perplexity == -7.1
        assert not hasattr(result, "model")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/test_topic_model.py::TestSweepResult -v`
Expected: FAIL — `SweepResult.__init__` requires `model` argument, and the dataclass has a `model` attribute

**Step 3: Remove `model` from `SweepResult` and update sweep function**

In `src/pipeline/topic_model.py`:

Replace `SweepResult` dataclass (lines 18-25):

```python
@dataclass
class SweepResult:
    """Result of a single LDA run during the parameter sweep."""

    k: int
    coherence: float
    perplexity: float
```

Update the `results.append` line (line 93):

```python
        results.append(SweepResult(k=k, coherence=score, perplexity=perplexity))
```

**Step 4: Run all tests**

Run: `cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy && PYTHONPATH=src pytest tests/ -v`
Expected: ALL PASS (check that existing sweep tests don't reference `result.model`)

> If any existing test accesses `result.model`, remove that assertion — models are no longer stored in sweep results.

**Step 5: Commit**

```bash
cd /Users/robsonguedes/py/slr-pipeline/.claude/worktrees/stoic-ptolemy
git add src/pipeline/topic_model.py tests/test_topic_model.py
git commit -m "refactor(topic_model): remove LdaModel from SweepResult

SweepResult now stores only (k, coherence, perplexity).
For K=2..15, this avoids holding 14 full LDA models in RAM.
The final model is trained separately at the optimal K."
```

---

## Task Dependency Graph

```
Task 1 (setup_nltk POS tagger) ── no deps
Task 2 (clean_text rewrite)    ── depends on Task 1
Task 3 (c_v coherence)         ── no deps
Task 4 (runner wiring)         ── depends on Task 3
Task 5 (SweepResult memory)    ── depends on Task 3
```

Parallelizable: Tasks 1+3 can run in parallel. Then Tasks 2+4+5 after their deps.
