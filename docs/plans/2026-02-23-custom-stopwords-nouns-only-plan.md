# Custom Stopwords & Nouns-Only Filter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add academic/domain-specific stopwords and a nouns-only POS filter to the preprocessing pipeline, configurable via PipelineConfig.

**Architecture:** Two config fields feed into `clean_text()` via new optional parameters. The runner threads them from config to the function call. No new modules needed — just config, preprocess, and runner modifications.

**Tech Stack:** NLTK (POS tagging already in place), Python dataclasses

---

## Task 1: Add stopword fields to PipelineConfig

**Files:**
- Modify: `src/pipeline/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
class TestPreprocessingDefaults:
    def test_academic_stopwords_has_defaults(self):
        config = PipelineConfig()
        assert isinstance(config.academic_stopwords, tuple)
        assert len(config.academic_stopwords) > 0
        assert "study" in config.academic_stopwords
        assert "research" in config.academic_stopwords

    def test_domain_stopwords_empty_by_default(self):
        config = PipelineConfig()
        assert config.domain_stopwords == ()

    def test_nouns_only_false_by_default(self):
        config = PipelineConfig()
        assert config.nouns_only is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py::TestPreprocessingDefaults -v`
Expected: AttributeError (fields don't exist yet)

**Step 3: Write minimal implementation**

Add to `src/pipeline/config.py`, after the recency section (line 87):

```python
    # ── Preprocessing (Step 4) ────────────────────────────────────
    academic_stopwords: tuple[str, ...] = (
        "study",
        "research",
        "result",
        "method",
        "approach",
        "analysis",
        "finding",
        "paper",
        "propose",
        "investigate",
        "examine",
        "evaluate",
        "demonstrate",
        "present",
        "discuss",
        "conclude",
        "suggest",
        "indicate",
        "reveal",
        "aim",
        "objective",
        "contribution",
        "limitation",
        "implication",
        "hypothesis",
        "conclusion",
        "introduction",
        "literature",
        "review",
        "methodology",
        "framework",
        "significant",
        "significantly",
        "respectively",
        "furthermore",
        "moreover",
        "however",
        "therefore",
        "consequently",
        "nevertheless",
        "whereas",
        "although",
    )
    domain_stopwords: tuple[str, ...] = ()
    nouns_only: bool = False
```

Update the class docstring to document the three new fields:

```python
    academic_stopwords : tuple[str, ...]
        Common academic terms removed during preprocessing. These words
        appear in most papers regardless of topic and add noise to LDA.
    domain_stopwords : tuple[str, ...]
        User-supplied domain-specific stopwords (empty by default).
        Set per-project for terms ubiquitous in your field.
    nouns_only : bool
        If True, keep only nouns after POS tagging (default: False).
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: All pass (including new tests)

**Step 5: Commit**

```bash
git add src/pipeline/config.py tests/test_config.py
git commit -m "feat(config): add academic_stopwords, domain_stopwords, and nouns_only fields

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Add extra_stopwords and nouns_only parameters to clean_text

**Files:**
- Modify: `src/pipeline/preprocess.py`
- Modify: `tests/test_preprocess.py`

**Step 1: Write the failing tests**

Add to `tests/test_preprocess.py`:

```python
class TestCleanTextExtraStopwords:
    def test_removes_extra_stopwords(self):
        tokens = clean_text(
            "this study investigates machine learning methods",
            extra_stopwords={"study", "method"},
        )
        assert "study" not in tokens
        assert "method" not in tokens
        assert "machine" in tokens

    def test_empty_extra_stopwords_changes_nothing(self):
        tokens_default = clean_text("machine learning algorithms")
        tokens_explicit = clean_text(
            "machine learning algorithms", extra_stopwords=set()
        )
        assert tokens_default == tokens_explicit


class TestCleanTextNounsOnly:
    def test_keeps_only_nouns(self):
        # "running" is a verb, "algorithm" is a noun
        tokens = clean_text("running the algorithm quickly", nouns_only=True)
        assert "algorithm" in tokens
        assert "running" not in tokens
        assert "quickly" not in tokens

    def test_nouns_only_false_keeps_all_pos(self):
        tokens = clean_text("running the algorithm quickly", nouns_only=False)
        # "run" (lemmatized verb) should be present when nouns_only=False
        assert "algorithm" in tokens
        assert "run" in tokens
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py::TestCleanTextExtraStopwords tests/test_preprocess.py::TestCleanTextNounsOnly -v`
Expected: TypeError (unexpected keyword arguments)

**Step 3: Write minimal implementation**

Modify `clean_text()` in `src/pipeline/preprocess.py`:

```python
def clean_text(
    text: str,
    extra_stopwords: set[str] | None = None,
    nouns_only: bool = False,
) -> list[str]:
    """Clean, tokenize, POS-lemmatize, and deduplicate a single document.

    Pipeline (matches legacy notebook methodology):
    1. Lowercase & unidecode (remove accents)
    2. Tokenize with RegexpTokenizer(r'\\w+')
    3. Remove stopwords (English + extra)
    4. POS-tag and lemmatize with correct part-of-speech
    5. Filter by POS if nouns_only is True
    6. Remove duplicate tokens (preserve first-occurrence order)
    7. Remove short tokens (len <= 2)

    Parameters
    ----------
    text : str
        Raw text (e.g. abstract + title).
    extra_stopwords : set[str] | None
        Additional stopwords to remove (academic + domain-specific).
    nouns_only : bool
        If True, keep only tokens tagged as nouns (NN, NNS, NNP, NNPS).

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

    # 3. Stopwords (English + extra)
    stops = set(stopwords.words("english"))
    if extra_stopwords:
        stops |= extra_stopwords
    tokens = [t for t in tokens if t not in stops]

    # 4. POS-aware lemmatization
    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(tokens)

    # 5. POS filter (before lemmatization, using original POS tags)
    if nouns_only:
        pos_tags = [(token, tag) for token, tag in pos_tags if tag.startswith("N")]

    tokens = [
        lemmatizer.lemmatize(token, _get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]

    # 6. Deduplicate (preserve order)
    tokens = list(OrderedDict.fromkeys(tokens))

    # 7. Length filter
    tokens = [t for t in tokens if len(t) > 2]

    return tokens
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: All pass (existing + new tests)

**Step 5: Commit**

```bash
git add src/pipeline/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): add extra_stopwords and nouns_only params to clean_text

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Thread config through the runner

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/test_runner.py`

**Step 1: Write the failing test**

Add to `tests/test_runner.py`:

```python
def test_passes_stopwords_and_nouns_only_to_clean_text(config_with_tempdir, monkeypatch):
    """Verify runner passes config stopwords and nouns_only to clean_text."""
    cfg = config_with_tempdir
    cfg.academic_stopwords = ("study",)
    cfg.domain_stopwords = ("blockchain",)
    cfg.nouns_only = True

    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    captured_kwargs = {}
    original_clean = clean_text

    def spy_clean(text, **kwargs):
        captured_kwargs.update(kwargs)
        return original_clean(text, **kwargs)

    monkeypatch.setattr("pipeline.runner.clean_text", spy_clean)

    run_pipeline(cfg)

    assert "extra_stopwords" in captured_kwargs
    assert "study" in captured_kwargs["extra_stopwords"]
    assert "blockchain" in captured_kwargs["extra_stopwords"]
    assert captured_kwargs["nouns_only"] is True
```

Note: This test uses `monkeypatch` to spy on `clean_text` without fully mocking it. You may need to mock other expensive parts (LDA sweep, plotting) to keep it fast. Follow the same mocking pattern as existing runner tests.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py::test_passes_stopwords_and_nouns_only_to_clean_text -v`
Expected: Fail (clean_text not called with extra_stopwords/nouns_only)

**Step 3: Write minimal implementation**

Modify runner.py line 79 — change:

```python
    tokens = [clean_text(doc) for doc in text_data]
```

to:

```python
    extra_stopwords = set(config.academic_stopwords) | set(config.domain_stopwords)
    tokens = [
        clean_text(doc, extra_stopwords=extra_stopwords, nouns_only=config.nouns_only)
        for doc in text_data
    ]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: All pass

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 131+ tests pass

**Step 6: Commit**

```bash
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(runner): pass academic/domain stopwords and nouns_only to clean_text

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task Dependency Graph

```
Task 1 (config fields)
    └── Task 2 (clean_text params)  ── can run independently
            └── Task 3 (runner threading)  ── depends on both
```

Tasks 1 and 2 are independent. Task 3 depends on both.

## Design Rationale

**Why config.py?** All pipeline knobs live there. Keeping stopwords in config means the exact preprocessing parameters are captured alongside thresholds, topic ranges, and other settings — essential for reproducible research.

**Why two separate fields?** Academic stopwords have sensible defaults (common across all SLR projects). Domain stopwords are empty by default and set per-project. This separation lets users extend academic defaults without losing them.

**Why `nouns_only` as a bool in config?** It's a preprocessing decision that affects topic quality and should be reproducible. Putting it in config makes it explicit and auditable.

**Why filter POS before lemmatization?** POS tags are more accurate on the original word forms. Filtering after lemmatization would miss verbs that lemmatize to noun-like forms.
