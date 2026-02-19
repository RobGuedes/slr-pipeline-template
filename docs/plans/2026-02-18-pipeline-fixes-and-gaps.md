# Pipeline Fixes & Architectural Gaps — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all confirmed issues from the audit — bugs, architectural gaps, missing persistence, resume capability, PRISMA tracking, dependency hygiene, and the missing scientometric module.

**Architecture:** Each task is self-contained and can be committed independently. Tasks are ordered by dependency: config changes first (other tasks depend on them), then module-level fixes, then the new scientometric module last. TDD throughout — every change starts with a failing test.

**Tech Stack:** Python 3.13, pandas, gensim, pytest, ruff, mypy

---

## Task 1: Fix `pyproject.toml` version targets and missing dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update ruff and mypy targets to match requires-python**

In `pyproject.toml`, change:
```toml
[tool.ruff]
target-version = "py313"

[tool.mypy]
python_version = "3.13"
```

**Step 2: Add missing runtime dependencies**

Add to `[project.dependencies]`:
```toml
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "nltk>=3.8.0",
    "scikit-learn>=1.3.0",
    "gensim>=4.3.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "unidecode>=1.3.0",
    "tqdm>=4.60.0",
    "openpyxl>=3.1.0",
]
```

Move `pyLDAvis`, `litstudy`, and `wordcloud` to optional:
```toml
[project.optional-dependencies]
viz = [
    "pyLDAvis>=3.4.0",
    "litstudy>=1.0.6",
    "wordcloud>=1.9.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
    "mypy",
]
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "fix: align ruff/mypy targets to py313, add missing deps"
```

---

## Task 2: Add `coherence_metric` to `PipelineConfig`

**Files:**
- Modify: `src/pipeline/config.py`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

```python
def test_coherence_metric_default():
    config = PipelineConfig()
    assert config.coherence_metric == "u_mass"

def test_coherence_metric_custom():
    config = PipelineConfig(coherence_metric="c_v")
    assert config.coherence_metric == "c_v"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_coherence_metric_default -v`
Expected: FAIL — `PipelineConfig` has no `coherence_metric` field

**Step 3: Add the field to PipelineConfig**

In `config.py`, add after `min_citations_per_topic`:
```python
    # ── Coherence metric ───────────────────────────────────────────
    coherence_metric: str = "u_mass"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Update `topic_model.py` to use config's coherence metric**

This is done in Task 3 (sweep refactor).

**Step 6: Commit**

```bash
git add src/pipeline/config.py tests/test_config.py
git commit -m "feat(config): add coherence_metric parameter"
```

---

## Task 3: Refactor `SweepResult` — stop storing full models in memory

**Files:**
- Modify: `src/pipeline/topic_model.py`
- Modify: `tests/test_topic_model.py`
- Modify: `src/pipeline/synthesis.py` (import of `SweepResult` for `plot_topic_audit`)

**Step 1: Write the failing test**

```python
def test_sweep_result_has_no_model_field():
    """SweepResult should store metrics only, not full LDA models."""
    from pipeline.topic_model import SweepResult
    result = SweepResult(k=5, coherence=0.42, perplexity=-7.1)
    assert not hasattr(result, "model")
    assert result.k == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_model.py::test_sweep_result_has_no_model_field -v`
Expected: FAIL — `SweepResult` requires `model` argument

**Step 3: Remove `model` from `SweepResult`**

```python
@dataclass
class SweepResult:
    """Result of a single LDA run during the parameter sweep."""
    k: int
    coherence: float
    perplexity: float
```

Update `perform_lda_sweep` to not store the model:
```python
results.append(SweepResult(k=k, coherence=score, perplexity=perplexity))
```

**Step 4: Fix any existing tests that construct `SweepResult` with a `model` argument**

Update test fixtures to remove the `model` field.

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/pipeline/topic_model.py tests/test_topic_model.py
git commit -m "refactor(topic_model): remove full LdaModel from SweepResult to save memory"
```

---

## Task 4: Add model persistence (save LDA model, dictionary, corpus)

**Files:**
- Create: `src/pipeline/persistence.py`
- Create: `tests/test_persistence.py`
- Modify: `src/pipeline/runner.py`

**Step 1: Write the failing test**

```python
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from pipeline.persistence import save_artifacts, load_artifacts


def test_save_and_load_model(tmp_path):
    """Round-trip: save model+dict+corpus, reload, verify."""
    # We'll use a mock for LdaModel since Gensim save/load needs actual training
    # Better: use a tiny real model from fixture
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    texts = [["hello", "world"], ["world", "test"]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=1, random_state=42)

    save_artifacts(model, dictionary, corpus, tmp_path)

    assert (tmp_path / "lda_model").exists()
    assert (tmp_path / "dictionary.dict").exists()
    assert (tmp_path / "corpus.mm").exists()

    model2, dict2, corpus2 = load_artifacts(tmp_path)
    assert model2.num_topics == 2
    assert len(dict2) == len(dictionary)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_persistence.py -v`
Expected: FAIL — module does not exist

**Step 3: Implement `persistence.py`**

```python
"""Model and artifact persistence for the SLR pipeline."""

from __future__ import annotations

from pathlib import Path

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel


def save_artifacts(
    model: LdaModel,
    dictionary: Dictionary,
    corpus: list[list[tuple[int, int]]],
    output_dir: Path,
) -> None:
    """Save LDA model, dictionary, and corpus to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(output_dir / "lda_model"))
    dictionary.save(str(output_dir / "dictionary.dict"))
    MmCorpus.serialize(str(output_dir / "corpus.mm"), corpus)


def load_artifacts(
    input_dir: Path,
) -> tuple[LdaModel, Dictionary, list]:
    """Load previously saved LDA model, dictionary, and corpus."""
    input_dir = Path(input_dir)

    model = LdaModel.load(str(input_dir / "lda_model"))
    dictionary = Dictionary.load(str(input_dir / "dictionary.dict"))
    corpus = list(MmCorpus(str(input_dir / "corpus.mm")))

    return model, dictionary, corpus
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_persistence.py -v`
Expected: PASS

**Step 5: Wire into `runner.py`**

Add after final model training (after line 105):
```python
from pipeline.persistence import save_artifacts
# ... after train_final_model:
save_artifacts(final_model, dictionary, corpus, config.processed_dir)
logger.info(f"Saved model artifacts to {config.processed_dir}")
```

**Step 6: Commit**

```bash
git add src/pipeline/persistence.py tests/test_persistence.py src/pipeline/runner.py
git commit -m "feat: add model persistence — save/load LDA model, dictionary, corpus"
```

---

## Task 5: Add PRISMA flow tracking

**Files:**
- Create: `src/pipeline/prisma.py`
- Create: `tests/test_prisma.py`
- Modify: `src/pipeline/runner.py`

**Step 1: Write the failing test**

```python
from pipeline.prisma import PrismaTracker


def test_prisma_tracker_records_counts():
    tracker = PrismaTracker()
    tracker.record("identified", 500)
    tracker.record("after_dedup", 420)
    tracker.record("after_type_filter", 380)
    tracker.record("after_topic_filter", 120)
    tracker.record("after_quality_review", 95)

    counts = tracker.counts
    assert counts["identified"] == 500
    assert counts["after_quality_review"] == 95


def test_prisma_tracker_export(tmp_path):
    tracker = PrismaTracker()
    tracker.record("identified", 500)
    tracker.record("after_dedup", 420)

    path = tmp_path / "prisma.csv"
    tracker.export_csv(path)

    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) == 2
    assert df.iloc[0]["stage"] == "identified"
    assert df.iloc[0]["count"] == 500
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prisma.py -v`
Expected: FAIL — module does not exist

**Step 3: Implement `prisma.py`**

```python
"""PRISMA flow tracking for SLR pipeline stages."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd


class PrismaTracker:
    """Tracks document counts at each pipeline stage for PRISMA reporting."""

    def __init__(self) -> None:
        self._counts: OrderedDict[str, int] = OrderedDict()

    def record(self, stage: str, count: int) -> None:
        """Record the document count at a given pipeline stage."""
        self._counts[stage] = count

    @property
    def counts(self) -> dict[str, int]:
        """Return all recorded counts."""
        return dict(self._counts)

    def export_csv(self, path: Path | str) -> None:
        """Export PRISMA counts to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            list(self._counts.items()), columns=["stage", "count"]
        )
        df.to_csv(path, index=False)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prisma.py -v`
Expected: PASS

**Step 5: Wire into `runner.py`**

Add `PrismaTracker` instantiation at the start of `run_pipeline()` and `tracker.record()` calls after each stage:
```python
from pipeline.prisma import PrismaTracker

tracker = PrismaTracker()
# After ingest:
tracker.record("identified", len(df_raw))
# After topic filter:
tracker.record("after_topic_filter", len(df_selected))
# At end:
tracker.export_csv(config.processed_dir / "prisma_counts.csv")
```

**Step 6: Commit**

```bash
git add src/pipeline/prisma.py tests/test_prisma.py src/pipeline/runner.py
git commit -m "feat: add PRISMA flow tracking across pipeline stages"
```

---

## Task 6: Add pipeline resume capability (two-phase runner)

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/test_runner.py`

**Step 1: Write the failing test**

```python
def test_run_pipeline_post_review_phase(tmp_path):
    """Runner should accept phase='post_review' and use load_reviewed."""
    # Create a mock reviewed CSV
    import pandas as pd
    review_path = tmp_path / "to_review.csv"
    df = pd.DataFrame({
        "title": ["Paper A", "Paper B"],
        "abstract": ["abs a", "abs b"],
        "author": ["Auth A", "Auth B"],
        "year": [2020, 2021],
        "cited_by": [10, 20],
        "Dominant_Topic": [0, 1],
        "Perc_Contribution": [0.8, 0.9],
        "Keep": ["yes", "no"],
    })
    df.to_csv(review_path, index=False)

    from pipeline.quality_review import load_reviewed
    kept = load_reviewed(review_path)
    assert len(kept) == 1
    assert kept.iloc[0]["title"] == "Paper A"
```

**Step 2: Design the two-phase approach**

Add a `phase` parameter to `run_pipeline`:
- `phase="full"` (default): runs everything, stops after export_for_review
- `phase="post_review"`: loads reviewed CSV, runs synthesis on kept papers only

**Step 3: Implement**

In `runner.py`, import `load_reviewed` and add phase logic:
```python
from pipeline.quality_review import export_for_review, load_reviewed

def run_pipeline(config: PipelineConfig | None = None, phase: str = "full") -> None:
    # ... existing code for full phase ...

    if phase == "full":
        export_for_review(df_selected, review_path)
        logger.info("Pipeline paused. Review 'to_review.csv' then re-run with phase='post_review'.")
        return

    if phase == "post_review":
        review_path = config.processed_dir / "to_review.csv"
        df_reviewed = load_reviewed(review_path)
        logger.info(f"Loaded {len(df_reviewed)} reviewed papers.")
        # Continue to synthesis with df_reviewed instead of df_selected
```

**Step 4: Run tests**

Run: `pytest tests/test_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(runner): add two-phase execution with post-review resume"
```

---

## Task 7: Fix `synthesis.py` — lazy imports for optional deps

**Files:**
- Modify: `src/pipeline/synthesis.py`
- Run: `pytest tests/test_synthesis.py -v`

**Step 1: Move top-level imports of pyLDAvis and litstudy to inside functions**

Replace lines 12-14:
```python
# Remove top-level:
# import pyLDAvis
# import pyLDAvis.gensim_models
# import litstudy
```

Move into each function that uses them:
- `convert_to_litstudy()`: `import litstudy` inside function body
- `plot_topics()`: `import pyLDAvis; import pyLDAvis.gensim_models` inside function body
- `plot_bibliometrics()`: `import litstudy` inside function body

**Step 2: Improve tempfile pattern**

Replace the `os.remove` with `Path.unlink(missing_ok=True)`:
```python
finally:
    Path(tmp_path).unlink(missing_ok=True)
```

And remove `import os` from the function since it's no longer needed.

**Step 3: Run tests**

Run: `pytest tests/test_synthesis.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/pipeline/synthesis.py
git commit -m "fix(synthesis): lazy-import optional deps, improve tempfile cleanup"
```

---

## Task 8: Remove dead code in `topic_select.py`

**Files:**
- Modify: `src/pipeline/topic_select.py`

**Step 1: Remove unused `present_topics` variable**

Delete line 113:
```python
present_topics = df["Dominant_Topic"].unique()
```

**Step 2: Run tests**

Run: `pytest tests/test_topic_select.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/pipeline/topic_select.py
git commit -m "fix(topic_select): remove unused present_topics variable"
```

---

## Task 9: Create `scientometric.py` module (Bradford, Lotka, keyword trends)

**Files:**
- Create: `src/pipeline/scientometric.py`
- Create: `tests/test_scientometric.py`

**Step 1: Write failing tests**

```python
import pandas as pd
from pipeline.scientometric import (
    bradford_zones,
    lotka_analysis,
    keyword_frequency,
)


def test_bradford_zones():
    df = pd.DataFrame({
        "source_title": ["J1"] * 50 + ["J2"] * 30 + ["J3"] * 10 + ["J4"] * 5 + ["J5"] * 3 + ["J6"] * 2,
    })
    zones = bradford_zones(df, zone_count=3)
    assert "zone" in zones.columns
    assert "source_title" in zones.columns
    assert set(zones["zone"].unique()) == {1, 2, 3}


def test_lotka_analysis():
    df = pd.DataFrame({
        "author": ["A; B", "A; C", "A", "D; B", "E"],
    })
    result = lotka_analysis(df)
    assert "author" in result.columns
    assert "paper_count" in result.columns
    # Author A appears in 3 papers
    a_row = result[result["author"] == "A"]
    assert a_row.iloc[0]["paper_count"] == 3


def test_keyword_frequency():
    df = pd.DataFrame({
        "author_keywords": ["ml; ai; deep learning", "ai; nlp", "ml; ai"],
    })
    result = keyword_frequency(df, top_n=3)
    assert result.iloc[0]["keyword"] == "ai"
    assert result.iloc[0]["count"] == 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scientometric.py -v`
Expected: FAIL — module does not exist

**Step 3: Implement `scientometric.py`**

```python
"""Scientometric analysis: Bradford zones, Lotka's law, keyword trends."""

from __future__ import annotations

import pandas as pd


def bradford_zones(
    df: pd.DataFrame,
    source_col: str = "source_title",
    zone_count: int = 3,
) -> pd.DataFrame:
    """Classify journals into Bradford zones.

    Zone 1 (core): fewest journals producing ~1/3 of papers.
    Zone 2 (related): next set producing ~1/3.
    Zone 3 (peripheral): remaining journals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a source/journal column.
    source_col : str
        Column name for journal/source title.
    zone_count : int
        Number of zones (typically 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: source_title, paper_count, cumulative, zone.
    """
    counts = (
        df[source_col]
        .value_counts()
        .reset_index()
    )
    counts.columns = [source_col, "paper_count"]
    counts["cumulative"] = counts["paper_count"].cumsum()
    total = counts["paper_count"].sum()

    boundaries = [total * (i + 1) / zone_count for i in range(zone_count)]
    zones = []
    for cum in counts["cumulative"]:
        for z, boundary in enumerate(boundaries, start=1):
            if cum <= boundary:
                zones.append(z)
                break
        else:
            zones.append(zone_count)
    counts["zone"] = zones

    return counts


def lotka_analysis(
    df: pd.DataFrame,
    author_col: str = "author",
    sep: str = ";",
) -> pd.DataFrame:
    """Compute author productivity (Lotka's law analysis).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an author column (semicolon-separated).
    author_col : str
        Column name for authors.
    sep : str
        Separator between author names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: author, paper_count — sorted descending.
    """
    authors: list[str] = []
    for raw in df[author_col].dropna():
        for name in str(raw).split(sep):
            name = name.strip()
            if name:
                authors.append(name)

    result = (
        pd.Series(authors)
        .value_counts()
        .reset_index()
    )
    result.columns = ["author", "paper_count"]
    return result


def keyword_frequency(
    df: pd.DataFrame,
    keyword_col: str = "author_keywords",
    sep: str = ";",
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute keyword frequency from author keywords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a keyword column (semicolon-separated).
    keyword_col : str
        Column name for keywords.
    sep : str
        Separator between keywords.
    top_n : int
        Number of top keywords to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: keyword, count — sorted descending.
    """
    keywords: list[str] = []
    for raw in df[keyword_col].dropna():
        for kw in str(raw).split(sep):
            kw = kw.strip().lower()
            if kw:
                keywords.append(kw)

    result = (
        pd.Series(keywords)
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    result.columns = ["keyword", "count"]
    return result
```

**Step 4: Run tests**

Run: `pytest tests/test_scientometric.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/scientometric.py tests/test_scientometric.py
git commit -m "feat: add scientometric module (Bradford zones, Lotka, keyword freq)"
```

---

## Task Dependency Graph

```
Task 1 (pyproject.toml)  ─── no deps, do first
Task 2 (config)          ─── no deps
Task 3 (SweepResult)     ─── no deps
Task 4 (persistence)     ─── after Task 3 (SweepResult no longer has model)
Task 5 (PRISMA)          ─── no deps
Task 6 (resume)          ─── after Task 5 (runner uses tracker)
Task 7 (synthesis lazy)  ─── no deps
Task 8 (dead code)       ─── no deps
Task 9 (scientometric)   ─── no deps, can be parallel
```

Tasks 1, 2, 3, 5, 7, 8, 9 are independent and can be parallelized.
Task 4 depends on Task 3. Task 6 depends on Task 5.
