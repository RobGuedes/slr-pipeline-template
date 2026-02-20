# Recency-Aware Citation Filter & Unified Plot Identity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add age-aware citation recovery to the topic selection filter, unify all bibliometric plots under a consistent style, and remove the litstudy plotting dependency.

**Architecture:** Two-pass citation filtering (strict filter → recovery pass). Shared matplotlib helper for uniform bar charts. All litstudy plot calls replaced with custom implementations. Top-15 author/source relevance computed from the full dataset before any filtering.

**Tech Stack:** pandas, matplotlib (Agg backend), dataclasses

---

## Task 1: Add recency config fields to `PipelineConfig`

**Files:**
- Modify: `src/pipeline/config.py:60-75`
- Test: `tests/test_config.py` (create if absent)

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for pipeline.config — configuration defaults and helpers."""

import pytest
from pipeline.config import PipelineConfig


class TestRecencyDefaults:
    def test_recency_filter_enabled_by_default(self):
        cfg = PipelineConfig()
        assert cfg.recency_filter_enabled is True

    def test_recent_threshold_years_default(self):
        cfg = PipelineConfig()
        assert cfg.recent_threshold_years == 2

    def test_mid_range_threshold_years_default(self):
        cfg = PipelineConfig()
        assert cfg.mid_range_threshold_years == 6

    def test_mid_range_min_citations_default(self):
        cfg = PipelineConfig()
        assert cfg.mid_range_min_citations == 5

    def test_top_n_authors_default(self):
        cfg = PipelineConfig()
        assert cfg.top_n_authors == 15

    def test_top_n_sources_default(self):
        cfg = PipelineConfig()
        assert cfg.top_n_sources == 15

    def test_reference_year_none_by_default(self):
        cfg = PipelineConfig()
        assert cfg.reference_year is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — attributes not found on `PipelineConfig`

**Step 3: Write minimal implementation**

Add these fields to `PipelineConfig` in `src/pipeline/config.py`, after line 63 (`min_citations_per_topic`):

```python
    # ── Recency-aware citation recovery (Step 7) ────────────────
    recency_filter_enabled: bool = True
    recent_threshold_years: int = 2
    mid_range_threshold_years: int = 6
    mid_range_min_citations: int = 5
    top_n_authors: int = 15
    top_n_sources: int = 15
    reference_year: int | None = None
```

Update the class docstring to document these new fields.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all 7 tests green)

**Step 5: Run full test suite for regression**

Run: `pytest --tb=short -q`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/pipeline/config.py tests/test_config.py
git commit -m "feat(config): add recency-aware citation filter fields"
```

---

## Task 2: Add `compute_top_authors` and `compute_top_sources` helpers

**Files:**
- Modify: `src/pipeline/topic_select.py` (add two new functions)
- Modify: `tests/test_topic_select.py` (add new test class)

**Step 1: Write the failing tests**

Add to `tests/test_topic_select.py`:

```python
from pipeline.topic_select import compute_top_authors, compute_top_sources


class TestComputeTopAuthors:
    def test_returns_top_n_by_paper_count(self):
        df = pd.DataFrame({
            "author": [
                "Smith, J.; Doe, A.",
                "Smith, J.; Brown, B.",
                "Doe, A.; Brown, B.",
                "Smith, J.",
            ]
        })
        top = compute_top_authors(df, n=2)
        # Smith appears in 3 papers, Doe in 2, Brown in 2
        assert top[0] == "Smith, J."
        assert len(top) == 2

    def test_handles_missing_author_column(self):
        df = pd.DataFrame({"title": ["A"]})
        top = compute_top_authors(df, n=5)
        assert top == []

    def test_handles_nan_authors(self):
        df = pd.DataFrame({"author": [None, "Smith, J."]})
        top = compute_top_authors(df, n=5)
        assert "Smith, J." in top


class TestComputeTopSources:
    def test_returns_top_n_by_paper_count(self):
        df = pd.DataFrame({
            "source_title": ["Journal A", "Journal A", "Journal B", "Journal C"]
        })
        top = compute_top_sources(df, n=2)
        assert top[0] == "Journal A"
        assert len(top) == 2

    def test_handles_missing_source_column(self):
        df = pd.DataFrame({"title": ["A"]})
        top = compute_top_sources(df, n=5)
        assert top == []

    def test_handles_nan_sources(self):
        df = pd.DataFrame({"source_title": [None, "J1", "J1"]})
        top = compute_top_sources(df, n=5)
        assert "J1" in top
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_select.py::TestComputeTopAuthors tests/test_topic_select.py::TestComputeTopSources -v`
Expected: FAIL — `ImportError: cannot import name 'compute_top_authors'`

**Step 3: Write minimal implementation**

Add to `src/pipeline/topic_select.py` (before `filter_documents`):

```python
def compute_top_authors(df: pd.DataFrame, n: int = 15) -> list[str]:
    """Return the top *n* authors by paper count from the full dataset.

    Authors are semicolon-separated in the ``author`` column.
    Each author is stripped and counted individually.
    """
    if "author" not in df.columns:
        return []

    authors: list[str] = []
    for raw in df["author"].dropna():
        for name in str(raw).split(";"):
            name = name.strip()
            if name:
                authors.append(name)

    if not authors:
        return []

    counts = pd.Series(authors).value_counts()
    return counts.head(n).index.tolist()


def compute_top_sources(df: pd.DataFrame, n: int = 15) -> list[str]:
    """Return the top *n* publication sources by paper count.

    Uses the ``source_title`` column directly.
    """
    if "source_title" not in df.columns:
        return []

    counts = df["source_title"].dropna().value_counts()
    return counts.head(n).index.tolist()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_select.py::TestComputeTopAuthors tests/test_topic_select.py::TestComputeTopSources -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/topic_select.py tests/test_topic_select.py
git commit -m "feat(topic_select): add compute_top_authors and compute_top_sources"
```

---

## Task 3: Add `recover_recent_papers` function

**Files:**
- Modify: `src/pipeline/topic_select.py` (add function)
- Modify: `tests/test_topic_select.py` (add test class)

**Step 1: Write the failing tests**

Add to `tests/test_topic_select.py`:

```python
from pipeline.topic_select import recover_recent_papers


class TestRecoverRecentPapers:
    """Test the second-pass recency recovery logic."""

    def _make_config(self, **overrides):
        return PipelineConfig(
            min_topic_prob=0.0,
            min_citations=10,
            recency_filter_enabled=True,
            recent_threshold_years=2,
            mid_range_threshold_years=6,
            mid_range_min_citations=5,
            reference_year=2026,
            **overrides,
        )

    def test_recovers_recent_paper_by_top_author(self):
        """< 2 years, 0 citations, author in top-15 → recovered."""
        df = pd.DataFrame({
            "year": [2025],
            "cited_by": [0],
            "author": ["Smith, J."],
            "source_title": ["Obscure Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        top_authors = ["Smith, J."]
        top_sources = []
        result = recover_recent_papers(df, self._make_config(), top_authors, top_sources)
        assert len(result) == 1

    def test_recovers_recent_paper_by_top_source(self):
        """< 2 years, 0 citations, source in top-15 → recovered."""
        df = pd.DataFrame({
            "year": [2025],
            "cited_by": [0],
            "author": ["Nobody, X."],
            "source_title": ["Top Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        top_authors = []
        top_sources = ["Top Journal"]
        result = recover_recent_papers(df, self._make_config(), top_authors, top_sources)
        assert len(result) == 1

    def test_rejects_recent_paper_without_relevance(self):
        """< 2 years, 0 citations, neither top author nor source → not recovered."""
        df = pd.DataFrame({
            "year": [2025],
            "cited_by": [0],
            "author": ["Nobody, X."],
            "source_title": ["Obscure Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 0

    def test_recovers_mid_range_paper_with_enough_citations(self):
        """2-6 years, citations >= mid_range_min → recovered."""
        df = pd.DataFrame({
            "year": [2022],
            "cited_by": [5],
            "author": ["Nobody, X."],
            "source_title": ["Any Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 1

    def test_rejects_mid_range_paper_with_few_citations(self):
        """2-6 years, citations < mid_range_min → not recovered."""
        df = pd.DataFrame({
            "year": [2022],
            "cited_by": [2],
            "author": ["Nobody, X."],
            "source_title": ["Any Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), [], [])
        assert len(result) == 0

    def test_does_not_recover_old_papers(self):
        """≥ 6 years, even with many citations → not recovered (strict filter handles these)."""
        df = pd.DataFrame({
            "year": [2018],
            "cited_by": [50],
            "author": ["Smith, J."],
            "source_title": ["Top Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), ["Smith, J."], ["Top Journal"])
        assert len(result) == 0

    def test_handles_missing_year(self):
        """Papers with no year are not recovered."""
        df = pd.DataFrame({
            "year": [None],
            "cited_by": [0],
            "author": ["Smith, J."],
            "source_title": ["Top Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), ["Smith, J."], ["Top Journal"])
        assert len(result) == 0

    def test_multiauthor_match(self):
        """Any author matching top-15 qualifies the paper."""
        df = pd.DataFrame({
            "year": [2025],
            "cited_by": [0],
            "author": ["Nobody, X.; Smith, J.; Other, Y."],
            "source_title": ["Obscure Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        result = recover_recent_papers(df, self._make_config(), ["Smith, J."], [])
        assert len(result) == 1

    def test_disabled_returns_empty(self):
        """When recency_filter_enabled=False, nothing is recovered."""
        df = pd.DataFrame({
            "year": [2025],
            "cited_by": [0],
            "author": ["Smith, J."],
            "source_title": ["Top Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        cfg = self._make_config(recency_filter_enabled=False)
        result = recover_recent_papers(df, cfg, ["Smith, J."], ["Top Journal"])
        assert len(result) == 0

    def test_reference_year_defaults_to_current_year(self):
        """When reference_year is None, uses datetime.now().year."""
        from datetime import datetime
        current_year = datetime.now().year
        df = pd.DataFrame({
            "year": [current_year],
            "cited_by": [0],
            "author": ["Smith, J."],
            "source_title": ["Top Journal"],
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.8],
        })
        cfg = self._make_config(reference_year=None)
        result = recover_recent_papers(df, cfg, ["Smith, J."], ["Top Journal"])
        # age = 0, which is < 2 → recent bracket, top author → recovered
        assert len(result) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_select.py::TestRecoverRecentPapers -v`
Expected: FAIL — `ImportError: cannot import name 'recover_recent_papers'`

**Step 3: Write minimal implementation**

Add to `src/pipeline/topic_select.py`:

```python
from datetime import datetime


def recover_recent_papers(
    df: pd.DataFrame,
    config: "PipelineConfig",
    top_authors: list[str],
    top_sources: list[str],
) -> pd.DataFrame:
    """Recover recent papers that were rejected by the strict citation filter.

    Second pass of the two-pass filter. Scans rejected papers and returns
    those meeting age-based recency criteria.

    Age brackets:
    - < recent_threshold_years: 0 citations OK if author or source is relevant
    - recent_threshold_years to < mid_range_threshold_years: >= mid_range_min_citations
    - >= mid_range_threshold_years: not recovered (strict filter handles these)

    Parameters
    ----------
    df : pd.DataFrame
        Rejected papers (those that did NOT pass the strict filter).
    config : PipelineConfig
        Configuration with recency thresholds.
    top_authors : list[str]
        Top N authors by paper count (from full dataset).
    top_sources : list[str]
        Top N publication sources by paper count (from full dataset).

    Returns
    -------
    pd.DataFrame
        Papers recovered by recency criteria.
    """
    if not config.recency_filter_enabled or df.empty:
        return df.iloc[0:0].copy()

    ref_year = config.reference_year if config.reference_year is not None else datetime.now().year

    # Drop rows with missing year — cannot determine age
    has_year = df["year"].notna()
    df_valid = df[has_year].copy()
    if df_valid.empty:
        return df.iloc[0:0].copy()

    age = ref_year - df_valid["year"].astype(int)

    # Bracket 1: recent (< recent_threshold_years)
    recent_mask = age < config.recent_threshold_years
    if top_authors or top_sources:
        top_authors_set = set(top_authors)
        top_sources_set = set(top_sources)

        def _has_top_author(authors_str: str) -> bool:
            if pd.isna(authors_str):
                return False
            return any(
                a.strip() in top_authors_set
                for a in str(authors_str).split(";")
            )

        author_relevant = df_valid["author"].apply(_has_top_author) if top_authors_set else pd.Series(False, index=df_valid.index)
        source_relevant = df_valid["source_title"].isin(top_sources_set) if top_sources_set else pd.Series(False, index=df_valid.index)
        relevance_mask = author_relevant | source_relevant
    else:
        relevance_mask = pd.Series(False, index=df_valid.index)

    recent_recovered = recent_mask & relevance_mask

    # Bracket 2: mid-range (recent_threshold to < mid_range_threshold)
    mid_mask = (age >= config.recent_threshold_years) & (age < config.mid_range_threshold_years)
    mid_citation_ok = df_valid["cited_by"] >= config.mid_range_min_citations
    mid_recovered = mid_mask & mid_citation_ok

    # Union
    recovery_mask = recent_recovered | mid_recovered
    return df_valid[recovery_mask].copy()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_select.py::TestRecoverRecentPapers -v`
Expected: PASS (all 11 tests green)

**Step 5: Commit**

```bash
git add src/pipeline/topic_select.py tests/test_topic_select.py
git commit -m "feat(topic_select): add recover_recent_papers two-pass recovery"
```

---

## Task 4: Integrate recovery into `filter_documents` + cleanup

**Files:**
- Modify: `src/pipeline/topic_select.py:78-129` (update `filter_documents`)
- Modify: `tests/test_topic_select.py` (add integration test)

**Step 1: Write the failing test**

Add to `tests/test_topic_select.py`:

```python
class TestFilterDocumentsWithRecovery:
    """Integration test: strict filter + recency recovery."""

    def test_recovers_recent_paper_rejected_by_strict_filter(self):
        """A recent paper by a top author should survive the full filter."""
        df = pd.DataFrame({
            "Dominant_Topic": [0, 0, 0],
            "Perc_Contribution": [0.9, 0.9, 0.9],
            "cited_by": [50, 0, 3],
            "year": [2015, 2025, 2023],
            "author": ["Old, A.", "Smith, J.", "Mid, M."],
            "source_title": ["J1", "J2", "J3"],
        })
        # full_df has Smith in top authors
        full_df = pd.DataFrame({
            "author": ["Smith, J."] * 20 + ["Other, X."] * 5,
            "source_title": ["J2"] * 20 + ["J3"] * 5,
        })
        cfg = PipelineConfig(
            min_topic_prob=0.0,
            min_citations=10,
            recency_filter_enabled=True,
            reference_year=2026,
            mid_range_min_citations=5,
        )
        result = filter_documents(df, cfg, full_df=full_df)
        # Paper 0: 50 citations, old → passes strict filter
        # Paper 1: 0 citations, recent, top author → recovered
        # Paper 2: 3 citations, mid-range, < 5 → NOT recovered
        assert len(result) == 2
        assert set(result["author"]) == {"Old, A.", "Smith, J."}

    def test_backward_compatible_without_full_df(self):
        """When full_df is not provided, only strict filter applies (backward compat)."""
        df = pd.DataFrame({
            "Dominant_Topic": [0],
            "Perc_Contribution": [0.9],
            "cited_by": [0],
            "year": [2025],
            "author": ["Smith, J."],
            "source_title": ["J1"],
        })
        cfg = PipelineConfig(min_topic_prob=0.0, min_citations=10)
        result = filter_documents(df, cfg)
        assert len(result) == 0  # strict filter rejects, no recovery without full_df
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_select.py::TestFilterDocumentsWithRecovery -v`
Expected: FAIL — `filter_documents()` does not accept `full_df` parameter

**Step 3: Update `filter_documents`**

Modify `filter_documents` in `src/pipeline/topic_select.py`:

- Add optional `full_df: pd.DataFrame | None = None` parameter
- After the strict filter, if `config.recency_filter_enabled` and `full_df` is not None:
  - Compute `top_authors = compute_top_authors(full_df, config.top_n_authors)`
  - Compute `top_sources = compute_top_sources(full_df, config.top_n_sources)`
  - Get rejected papers: `df_rejected = df[~final_mask]`
  - Call `recovered = recover_recent_papers(df_rejected, config, top_authors, top_sources)`
  - Union: return `pd.concat([df[final_mask], recovered]).copy()`
- Remove the unused `present_topics` variable (line 113)
- Remove the uncertain comment on line 116, replace with definitive: `# Exclude documents that failed topic assignment`

The updated function signature:

```python
def filter_documents(
    df: pd.DataFrame,
    config: "PipelineConfig",
    full_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_select.py -v`
Expected: ALL tests pass (old + new)

**Step 5: Commit**

```bash
git add src/pipeline/topic_select.py tests/test_topic_select.py
git commit -m "feat(topic_select): integrate recency recovery into filter_documents

Removes unused present_topics variable. Adds optional full_df parameter
for computing top-15 authors/sources used in recency recovery."
```

---

## Task 5: Add shared `_plot_horizontal_bar` helper

**Files:**
- Modify: `src/pipeline/synthesis.py` (add helper function)
- Modify: `tests/test_synthesis.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_synthesis.py`:

```python
import tempfile
from pathlib import Path
from pipeline.synthesis import _plot_horizontal_bar
import pandas as pd


class TestPlotHorizontalBar:
    def test_creates_png_file(self):
        data = pd.Series({"A": 10, "B": 5, "C": 3})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.png"
            _plot_horizontal_bar(data, "No. of documents", "Test Title", out)
            assert out.exists()
            assert out.stat().st_size > 0

    def test_creates_parent_dirs(self):
        data = pd.Series({"A": 10})
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "nested" / "dir" / "test.png"
            _plot_horizontal_bar(data, "No. of documents", "Test Title", out)
            assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py::TestPlotHorizontalBar -v`
Expected: FAIL — cannot import `_plot_horizontal_bar`

**Step 3: Write minimal implementation**

Add to `src/pipeline/synthesis.py` (after the imports, before `generate_report`):

```python
def _plot_horizontal_bar(
    data: pd.Series,
    xlabel: str,
    title: str,
    output_path: Path | str,
) -> None:
    """Render a horizontal bar chart with unified project styling.

    Parameters
    ----------
    data : pd.Series
        Index = labels, values = counts. Assumed already sorted/truncated.
    xlabel : str
        Label for the x-axis.
    title : str
        Chart title.
    output_path : Path | str
        Where to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py::TestPlotHorizontalBar -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/synthesis.py tests/test_synthesis.py
git commit -m "feat(synthesis): add _plot_horizontal_bar shared styling helper"
```

---

## Task 6: Replace litstudy plots and add top sources plot

**Files:**
- Modify: `src/pipeline/synthesis.py:1-297` (rewrite `plot_bibliometrics`, remove litstudy)
- Modify: `tests/test_synthesis.py` (update/add plot tests)

**Step 1: Write the failing tests**

Add to `tests/test_synthesis.py`:

```python
from pipeline.synthesis import plot_bibliometrics


class TestPlotBibliometrics:
    @pytest.fixture
    def sample_biblio_df(self):
        return pd.DataFrame({
            "year": [2020, 2021, 2022, 2020, 2021],
            "author": [
                "Smith, J.; Doe, A.",
                "Smith, J.",
                "Brown, B.",
                "Doe, A.",
                "Smith, J.; Brown, B.",
            ],
            "source_title": ["J1", "J2", "J1", "J3", "J1"],
            "Affiliations": [
                "Dept A, Uni X, USA",
                "Dept B, Uni Y, UK",
                "Dept C, Uni X, USA",
                None,
                "Dept D, Uni Z, Brazil",
            ],
        })

    def test_creates_all_plot_files(self, sample_biblio_df):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_bibliometrics(sample_biblio_df, Path(tmpdir))
            assert (Path(tmpdir) / "publication_years.png").exists()
            assert (Path(tmpdir) / "top_authors.png").exists()
            assert (Path(tmpdir) / "top_sources.png").exists()
            assert (Path(tmpdir) / "top_countries.png").exists()
            assert (Path(tmpdir) / "top_affiliations.png").exists()

    def test_no_docs_parameter_required(self, sample_biblio_df):
        """Verify plot_bibliometrics no longer requires a DocumentSet."""
        import inspect
        sig = inspect.signature(plot_bibliometrics)
        param_names = list(sig.parameters.keys())
        assert "docs" not in param_names

    def test_handles_empty_dataframe(self):
        df = pd.DataFrame({
            "year": pd.Series(dtype=int),
            "author": pd.Series(dtype=str),
            "source_title": pd.Series(dtype=str),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            plot_bibliometrics(df, Path(tmpdir))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py::TestPlotBibliometrics -v`
Expected: FAIL — `plot_bibliometrics` still requires `docs` parameter

**Step 3: Rewrite `plot_bibliometrics`**

Replace the entire `plot_bibliometrics` function and remove litstudy imports:

1. Remove `import litstudy` (line 14)
2. Remove `from litstudy import DocumentSet` (line 19)
3. Remove `convert_to_litstudy` function (lines 22-60)
4. Change `plot_bibliometrics` signature to `(df: pd.DataFrame, output_dir: Path | str) -> None`
5. Replace the body:

```python
def plot_bibliometrics(
    df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Generate bibliometric plots and save them to output_dir.

    All plots use a unified visual identity. Horizontal bar charts use
    ``_plot_horizontal_bar``; the publication years chart uses vertical bars.

    Plots generated:
    - publication_years.png  (vertical bars)
    - top_authors.png        (horizontal bars)
    - top_sources.png        (horizontal bars)
    - top_countries.png      (horizontal bars)
    - top_affiliations.png   (horizontal bars)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Publication Years (vertical bars) ────────────────────────
    if "year" in df.columns and not df["year"].dropna().empty:
        year_counts = df["year"].dropna().astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        year_counts.plot.bar(ax=ax, color="#4c72b0")
        ax.set_xlabel("Year")
        ax.set_ylabel("No. of documents")
        ax.set_title("Publication Trends")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(out / "publication_years.png", dpi=150)
        plt.close(fig)

    # ── 2. Top Authors (horizontal bars) ────────────────────────────
    if "author" in df.columns:
        authors: list[str] = []
        for raw in df["author"].dropna():
            for name in str(raw).split(";"):
                name = name.strip()
                if name:
                    authors.append(name)
        if authors:
            author_counts = pd.Series(authors).value_counts().head(15)
            _plot_horizontal_bar(
                author_counts, "No. of documents", "Top 15 Authors",
                out / "top_authors.png",
            )

    # ── 3. Top Publication Sources (horizontal bars) ────────────────
    if "source_title" in df.columns:
        source_counts = df["source_title"].dropna().value_counts().head(15)
        if not source_counts.empty:
            _plot_horizontal_bar(
                source_counts, "No. of documents", "Top 15 Publication Sources",
                out / "top_sources.png",
            )

    # ── 4. Top Countries (horizontal bars) ──────────────────────────
    aff_col = "Affiliations" if "Affiliations" in df.columns else "affiliations"
    if aff_col in df.columns:
        countries: list[str] = []
        for raw in df[aff_col].dropna():
            for entry in str(raw).split(";"):
                parts = [p.strip() for p in entry.split(",")]
                if parts:
                    countries.append(parts[-1])
        if countries:
            country_counts = pd.Series(countries).value_counts().head(20)
            _plot_horizontal_bar(
                country_counts, "No. of documents", "Top 20 Countries",
                out / "top_countries.png",
            )

    # ── 5. Top Affiliations (horizontal bars) ───────────────────────
    if aff_col in df.columns:
        institutions: list[str] = []
        for raw in df[aff_col].dropna():
            for entry in str(raw).split(";"):
                parts = [p.strip() for p in entry.split(",")]
                inst = parts[1] if len(parts) >= 2 else parts[0] if parts else ""
                if inst:
                    institutions.append(inst)
        if institutions:
            inst_counts = pd.Series(institutions).value_counts().head(20)
            _plot_horizontal_bar(
                inst_counts, "No. of documents", "Top 20 Affiliations",
                out / "top_affiliations.png",
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/pipeline/synthesis.py tests/test_synthesis.py
git commit -m "feat(synthesis): replace litstudy plots with unified custom charts

Removes litstudy import, convert_to_litstudy(), and DocumentSet parameter.
Adds top-15 publication sources plot. All bar charts now use shared
_plot_horizontal_bar helper for consistent visual identity."
```

---

## Task 7: Update runner.py and clean up litstudy references

**Files:**
- Modify: `src/pipeline/runner.py:22-31, 158, 178-188`

**Step 1: Update imports and calls**

In `src/pipeline/runner.py`:

1. Remove `convert_to_litstudy` from the import (line 25)
2. Update `filter_documents` call (line 158) to pass `full_df=df_topics`:
   ```python
   df_selected = filter_documents(df_topics, config, full_df=df_topics)
   ```
   Note: `df_topics` is the full dataset with topic assignments. It serves as
   `full_df` because we want top-15 computed from ALL papers, not just selected ones.

3. Replace lines 178-188 (the litstudy conversion + plot_bibliometrics call):
   ```python
   # Generate bibliometric plots
   logger.info("Generating bibliometric plots...")
   plot_bibliometrics(df_selected, config.processed_dir)
   ```

4. Remove `convert_to_litstudy` from the import list (line 25)

**Step 2: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/pipeline/runner.py
git commit -m "refactor(runner): remove litstudy conversion, pass full_df to filter"
```

---

## Task 8: Update existing synthesis tests for new signature

**Files:**
- Modify: `tests/test_synthesis.py` (update `TestConvertToLitStudy`)

**Step 1: Remove or update obsolete tests**

- Delete `TestConvertToLitStudy` class entirely (tests a function that no longer exists)
- Update any remaining tests that reference `docs` parameter or litstudy mocks

**Step 2: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_synthesis.py
git commit -m "test(synthesis): remove obsolete convert_to_litstudy tests"
```

---

## Task 9: Final regression + cleanup

**Files:**
- All modified files

**Step 1: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

**Step 2: Run linting**

Run: `ruff check src/ tests/`
Expected: No errors (or only pre-existing ones)

**Step 3: Run formatter**

Run: `black --check src/ tests/`
Expected: No reformatting needed (or apply if needed)

**Step 4: Verify no litstudy references remain in src/pipeline/synthesis.py**

Run: `grep -n "litstudy" src/pipeline/synthesis.py`
Expected: No output

**Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: apply black formatting to modified files"
```
