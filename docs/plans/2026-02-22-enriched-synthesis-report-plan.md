# Enriched Synthesis Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive pipeline metrics tracking and generate both LaTeX and JSON synthesis reports showing data flow through all pipeline stages.

**Architecture:** Create a `PipelineMetrics` dataclass to accumulate statistics at each pipeline stage. Modify `filter_documents()` to expose recovery stats. Enhance `generate_report()` and `export_report_tex()` to use `PipelineMetrics`. Add new `export_report_json()` for machine-readable output.

**Tech Stack:** Python 3.13, pandas, dataclasses, JSON

---

## Task 1: Create PipelineMetrics dataclass

**Files:**
- Create: `src/pipeline/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
"""Tests for pipeline.metrics — pipeline statistics tracking."""

from pipeline.metrics import PipelineMetrics


class TestPipelineMetrics:
    def test_creates_instance_with_all_fields(self):
        metrics = PipelineMetrics(
            scopus_raw=450,
            wos_raw=320,
            duplicates_removed=85,
            unique_papers=685,
            selected_k=8,
            coherence_score=0.542,
            papers_per_topic={0: 85, 1: 72},
            failed_topic_assignment=12,
            passed_probability=520,
            passed_citations=420,
            papers_selected_strict=380,
            papers_recovered=35,
            papers_final=415,
            year_min=2016,
            year_max=2026,
            total_citations=7254,
        )
        assert metrics.scopus_raw == 450
        assert metrics.selected_k == 8
        assert metrics.papers_per_topic[0] == 85

    def test_to_dict_converts_to_structured_format(self):
        metrics = PipelineMetrics(
            scopus_raw=100, wos_raw=50, duplicates_removed=10, unique_papers=140,
            selected_k=5, coherence_score=0.5, papers_per_topic={0: 20},
            failed_topic_assignment=5, passed_probability=100, passed_citations=90,
            papers_selected_strict=80, papers_recovered=10, papers_final=90,
            year_min=2020, year_max=2025, total_citations=1000,
        )
        d = metrics.to_dict()
        assert d["ingestion"]["scopus_raw"] == 100
        assert d["topic_modeling"]["selected_k"] == 5
        assert d["filtering"]["papers_final"] == 90
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: ImportError (module doesn't exist)

**Step 3: Write minimal implementation**

Create `src/pipeline/metrics.py`:

```python
"""Pipeline metrics dataclass for synthesis reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline statistics for synthesis reporting.

    Tracks metrics across all pipeline stages: ingestion, topic modeling,
    filtering, and final dataset characteristics.
    """

    # Ingestion
    scopus_raw: int
    wos_raw: int
    duplicates_removed: int
    unique_papers: int

    # Topic modeling
    selected_k: int
    coherence_score: float
    papers_per_topic: dict[int, int]

    # Filtering
    failed_topic_assignment: int
    passed_probability: int
    passed_citations: int
    papers_selected_strict: int
    papers_recovered: int
    papers_final: int

    # Final stats
    year_min: int
    year_max: int
    total_citations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to structured dictionary for JSON export.

        Returns
        -------
        dict
            Nested structure with sections: ingestion, topic_modeling,
            filtering, final_dataset.
        """
        return {
            "ingestion": {
                "scopus_raw": self.scopus_raw,
                "wos_raw": self.wos_raw,
                "duplicates_removed": self.duplicates_removed,
                "unique_papers": self.unique_papers,
            },
            "topic_modeling": {
                "selected_k": self.selected_k,
                "coherence_score": self.coherence_score,
                "papers_per_topic": self.papers_per_topic,
            },
            "filtering": {
                "failed_topic_assignment": self.failed_topic_assignment,
                "passed_probability": self.passed_probability,
                "passed_citations": self.passed_citations,
                "papers_selected_strict": self.papers_selected_strict,
                "papers_recovered": self.papers_recovered,
                "papers_final": self.papers_final,
            },
            "final_dataset": {
                "year_min": self.year_min,
                "year_max": self.year_max,
                "total_citations": self.total_citations,
            },
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/pipeline/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add PipelineMetrics dataclass for synthesis reporting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Modify ingest_all to return source counts

**Files:**
- Modify: `src/pipeline/ingest.py:ingest_all()`
- Modify: `tests/test_ingest.py`

**Step 1: Write the failing test**

Add to `tests/test_ingest.py::TestIngestAll`:

```python
def test_returns_source_counts_when_requested(self):
    """Verify ingest_all can return per-source counts."""
    config = PipelineConfig(
        raw_dir=Path("tests/fixtures"),
        included_doc_types=("Article",),
    )
    df, counts = ingest_all(config.raw_dir, config, return_counts=True)
    assert "scopus" in counts
    assert "wos" in counts
    assert "unique" in counts
    assert "duplicates_removed" in counts
    assert isinstance(counts["scopus"], int)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py::TestIngestAll::test_returns_source_counts_when_requested -v`
Expected: TypeError (got unexpected keyword argument 'return_counts')

**Step 3: Write minimal implementation**

Modify `src/pipeline/ingest.py::ingest_all()` signature and implementation:

```python
def ingest_all(
    raw_dir: Path,
    config: PipelineConfig,
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Ingest and deduplicate Scopus CSV and WoS TXT exports.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw data files.
    config : PipelineConfig
        Pipeline configuration (doc types, etc.).
    return_counts : bool
        If True, return (df, counts) where counts has per-source stats.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]
        Unified deduplicated DataFrame. If return_counts=True, returns
        (df, counts) where counts = {"scopus": int, "wos": int, "unique": int,
        "duplicates_removed": int}.
    """
    scopus_files = list(raw_dir.glob("*.csv"))
    wos_files = list(raw_dir.glob("*.txt"))

    dfs: list[pd.DataFrame] = []
    scopus_count = 0
    wos_count = 0

    for f in scopus_files:
        df_scopus = load_scopus_csv(f, config)
        scopus_count += len(df_scopus)
        dfs.append(df_scopus)

    for f in wos_files:
        df_wos = load_wos_txt(f, config)
        wos_count += len(df_wos)
        dfs.append(df_wos)

    if not dfs:
        empty = pd.DataFrame()
        if return_counts:
            return empty, {"scopus": 0, "wos": 0, "unique": 0, "duplicates_removed": 0}
        return empty

    merged = pd.concat(dfs, ignore_index=True)
    pre_dedup_count = len(merged)
    result = deduplicate(merged)
    unique_count = len(result)
    duplicates_removed = pre_dedup_count - unique_count

    if return_counts:
        counts = {
            "scopus": scopus_count,
            "wos": wos_count,
            "unique": unique_count,
            "duplicates_removed": duplicates_removed,
        }
        return result, counts

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py::TestIngestAll::test_returns_source_counts_when_requested -v`
Expected: 1 passed

**Step 5: Run full ingest test suite to ensure backward compatibility**

Run: `pytest tests/test_ingest.py -v`
Expected: All tests pass (no regressions from adding optional parameter)

**Step 6: Commit**

```bash
git add src/pipeline/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): add optional return_counts parameter to ingest_all

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Modify filter_documents to return filter stats

**Files:**
- Modify: `src/pipeline/topic_select.py:filter_documents()`
- Modify: `tests/test_topic_select.py`

**Step 1: Write the failing test**

Add to `tests/test_topic_select.py`:

```python
class TestFilterDocumentsWithStats:
    def test_returns_filter_stage_counts(self):
        """Verify filter_documents can return intermediate filter counts."""
        config = PipelineConfig(min_topic_prob=0.7, min_citations=5)
        df = pd.DataFrame({
            "Dominant_Topic": [0, 1, 2, -1, 0],
            "Perc_Contribution": [0.8, 0.6, 0.9, 0.0, 0.75],
            "cited_by": [10, 3, 8, 0, 12],
        })
        result, stats = filter_documents(df, config, return_stats=True)
        assert "failed_topic_assignment" in stats
        assert "passed_probability" in stats
        assert "passed_citations" in stats
        assert "selected_strict" in stats
        assert stats["failed_topic_assignment"] == 1  # Dominant_Topic == -1
        assert stats["selected_strict"] == 2  # prob>=0.7, citations>=5, valid topic
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_select.py::TestFilterDocumentsWithStats::test_returns_filter_stage_counts -v`
Expected: TypeError (got unexpected keyword argument 'return_stats')

**Step 3: Write minimal implementation**

Modify `src/pipeline/topic_select.py::filter_documents()`:

```python
def filter_documents(
    df: pd.DataFrame,
    config: "PipelineConfig",
    full_df: pd.DataFrame | None = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Filter documents based on probability and citation count.

    Rules:
    1. Perc_Contribution >= config.min_topic_prob
    2. cited_by >= config.effective_min_citations(Dominant_Topic)

    When *full_df* is provided and ``config.recency_filter_enabled`` is True,
    a second recovery pass runs on rejected papers using age-based criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with "Dominant_Topic" and "Perc_Contribution".
    config : PipelineConfig
        Configuration with thresholds.
    full_df : pd.DataFrame | None
        Full dataset (before any filtering) used to compute top-15 authors
        and sources for recency recovery. If None, recovery is skipped.
    return_stats : bool
        If True, return (df, stats) where stats contains filter stage counts.

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]
        Filtered DataFrame. If return_stats=True, returns (df, stats) where
        stats = {"failed_topic_assignment": int, "passed_probability": int,
        "passed_citations": int, "selected_strict": int, "recovered": int}.
    """
    if "Dominant_Topic" not in df.columns or "Perc_Contribution" not in df.columns:
        raise ValueError("DataFrame must have assigned dominant topics.")

    # 1. Filter by probability
    prob_mask = df["Perc_Contribution"] >= config.min_topic_prob

    # 2. Exclude documents that failed topic assignment
    valid_topic_mask = df["Dominant_Topic"] != -1

    # 3. Filter by citations (per-topic adaptive)
    min_citations = df["Dominant_Topic"].map(config.effective_min_citations)
    citation_mask = df["cited_by"] >= min_citations

    # Combine masks
    final_mask = prob_mask & valid_topic_mask & citation_mask

    # Collect stats if requested
    if return_stats:
        stats = {
            "failed_topic_assignment": int((df["Dominant_Topic"] == -1).sum()),
            "passed_probability": int(prob_mask.sum()),
            "passed_citations": int(citation_mask.sum()),
            "selected_strict": int(final_mask.sum()),
            "recovered": 0,
        }

    # 4. Optional second-pass recency recovery
    if config.recency_filter_enabled and full_df is not None:
        top_authors = compute_top_authors(full_df, config.top_n_authors)
        top_sources = compute_top_sources(full_df, config.top_n_sources)
        df_rejected = df[~final_mask]
        recovered = recover_recent_papers(df_rejected, config, top_authors, top_sources)
        if return_stats:
            stats["recovered"] = len(recovered)
        result = pd.concat([df[final_mask], recovered]).copy()
        if return_stats:
            return result, stats
        return result

    if return_stats:
        return df[final_mask].copy(), stats

    return df[final_mask].copy()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_select.py::TestFilterDocumentsWithStats::test_returns_filter_stage_counts -v`
Expected: 1 passed

**Step 5: Run full topic_select test suite**

Run: `pytest tests/test_topic_select.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/pipeline/topic_select.py tests/test_topic_select.py
git commit -m "feat(topic_select): add optional return_stats to filter_documents

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Enhance generate_report and export_report_tex

**Files:**
- Modify: `src/pipeline/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Step 1: Write the failing test**

Add to `tests/test_synthesis.py`:

```python
from pipeline.metrics import PipelineMetrics


class TestGenerateReportWithMetrics:
    def test_accepts_pipeline_metrics(self):
        """Verify generate_report works with PipelineMetrics."""
        metrics = PipelineMetrics(
            scopus_raw=100, wos_raw=50, duplicates_removed=10, unique_papers=140,
            selected_k=5, coherence_score=0.5, papers_per_topic={0: 20},
            failed_topic_assignment=5, passed_probability=100, passed_citations=90,
            papers_selected_strict=80, papers_recovered=10, papers_final=90,
            year_min=2020, year_max=2025, total_citations=1000,
        )
        stats = generate_report(metrics)
        assert stats["ingestion"]["scopus_raw"] == 100
        assert stats["topic_modeling"]["selected_k"] == 5
        assert stats["filtering"]["papers_final"] == 90


class TestExportReportTexWithMetrics:
    def test_exports_comprehensive_latex_table(self):
        """Verify export_report_tex formats PipelineMetrics as LaTeX."""
        metrics = PipelineMetrics(
            scopus_raw=450, wos_raw=320, duplicates_removed=85, unique_papers=685,
            selected_k=8, coherence_score=0.542, papers_per_topic={},
            failed_topic_assignment=12, passed_probability=520, passed_citations=420,
            papers_selected_strict=380, papers_recovered=35, papers_final=415,
            year_min=2016, year_max=2026, total_citations=7254,
        )
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.tex"
            export_report_tex(metrics, out)
            assert out.exists()
            content = out.read_text()
            assert "Scopus (raw)" in content
            assert "450" in content
            assert "Topics (K)" in content
            assert "8" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py::TestGenerateReportWithMetrics -v`
Expected: Fail (generate_report doesn't accept PipelineMetrics)

**Step 3: Write minimal implementation**

Modify `src/pipeline/synthesis.py`:

```python
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from pipeline.metrics import PipelineMetrics


def generate_report(source: pd.DataFrame | "PipelineMetrics") -> dict[str, Any]:
    """Calculate summary statistics for the final dataset.

    Parameters
    ----------
    source : pd.DataFrame | PipelineMetrics
        Either a filtered DataFrame (legacy) or a PipelineMetrics instance.

    Returns
    -------
    dict
        If DataFrame: simple stats dict (total_papers, year range, citations).
        If PipelineMetrics: full nested dict via to_dict().
    """
    # Handle PipelineMetrics
    if hasattr(source, "to_dict"):
        return source.to_dict()

    # Legacy DataFrame path
    df = source
    if len(df) == 0:
        return {
            "total_papers": 0,
            "min_year": None,
            "max_year": None,
            "total_citations": 0,
        }

    return {
        "total_papers": len(df),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "total_citations": int(df["cited_by"].sum()),
    }


def export_report_tex(source: dict[str, Any] | "PipelineMetrics", output_path: Path | str) -> None:
    """Format synthesis statistics as a LaTeX table and save to .tex file.

    Parameters
    ----------
    source : dict | PipelineMetrics
        Either a stats dict from generate_report or a PipelineMetrics instance.
    output_path : Path | str
        Path to save the .tex file.
    """
    # Convert PipelineMetrics to dict
    if hasattr(source, "to_dict"):
        stats = source.to_dict()
    else:
        stats = source

    # Check if comprehensive metrics or legacy simple stats
    if "ingestion" in stats:
        # Comprehensive format
        ing = stats["ingestion"]
        tm = stats["topic_modeling"]
        filt = stats["filtering"]
        final = stats["final_dataset"]

        year_range = f"{final['year_min']} -- {final['year_max']}"
        total_cites = final["total_citations"]

        tex_content = rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lr}}
\hline
\textbf{{Metric}} & \textbf{{Value}} \\
\hline
\multicolumn{{2}}{{c}}{{\textit{{Ingestion}}}} \\
Scopus (raw) & {ing["scopus_raw"]} \\
WoS (raw) & {ing["wos_raw"]} \\
Duplicates removed & {ing["duplicates_removed"]} \\
Unique papers & {ing["unique_papers"]} \\
\hline
\multicolumn{{2}}{{c}}{{\textit{{Topic Modeling}}}} \\
Topics (K) & {tm["selected_k"]} \\
Coherence & {tm["coherence_score"]:.3f} \\
\hline
\multicolumn{{2}}{{c}}{{\textit{{Filtering}}}} \\
Failed topic assignment & {filt["failed_topic_assignment"]} \\
Passed probability & {filt["passed_probability"]} \\
Passed citations & {filt["passed_citations"]} \\
Selected (strict) & {filt["papers_selected_strict"]} \\
Recovered (recency) & {filt["papers_recovered"]} \\
\textbf{{Final selected}} & \textbf{{{filt["papers_final"]}}} \\
\hline
\multicolumn{{2}}{{c}}{{\textit{{Final Dataset}}}} \\
Year range & {year_range} \\
Total citations & {total_cites:,} \\
\hline
\end{{tabular}}
\caption{{Synthesis Report Statistics}}
\label{{tab:synthesis_report}}
\end{{table}}
"""
    else:
        # Legacy simple format
        total_docs = stats.get("total_papers", 0)
        min_year = stats.get("min_year", "")
        max_year = stats.get("max_year", "")
        total_cites = stats.get("total_citations", 0)

        year_range = f"{min_year} -- {max_year}" if min_year and max_year else "N/A"

        tex_content = rf"""\begin{{table}}[h]
\centering
\begin{{tabular}}{{lr}}
\hline
\textbf{{Metric}} & \textbf{{Value}} \\
\hline
Total Documents & {total_docs} \\
Year Range & {year_range} \\
Total Citations & {total_cites} \\
\hline
\end{{tabular}}
\caption{{Synthesis Report Statistics}}
\label{{tab:synthesis_report}}
\end{{table}}
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(tex_content)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py::TestGenerateReportWithMetrics tests/test_synthesis.py::TestExportReportTexWithMetrics -v`
Expected: 2 passed

**Step 5: Run full synthesis test suite**

Run: `pytest tests/test_synthesis.py -v`
Expected: All tests pass (backward compatibility preserved)

**Step 6: Commit**

```bash
git add src/pipeline/synthesis.py tests/test_synthesis.py
git commit -m "feat(synthesis): enhance generate_report and export_report_tex for PipelineMetrics

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add export_report_json function

**Files:**
- Modify: `src/pipeline/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Step 1: Write the failing test**

Add to `tests/test_synthesis.py`:

```python
import json


class TestExportReportJson:
    def test_exports_json_metadata(self):
        """Verify export_report_json writes structured JSON."""
        metrics = PipelineMetrics(
            scopus_raw=100, wos_raw=50, duplicates_removed=10, unique_papers=140,
            selected_k=5, coherence_score=0.5, papers_per_topic={0: 20, 1: 30},
            failed_topic_assignment=5, passed_probability=100, passed_citations=90,
            papers_selected_strict=80, papers_recovered=10, papers_final=90,
            year_min=2020, year_max=2025, total_citations=1000,
        )
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "metadata.json"
            export_report_json(metrics, out)
            assert out.exists()
            with open(out) as f:
                data = json.load(f)
            assert data["ingestion"]["scopus_raw"] == 100
            assert data["topic_modeling"]["papers_per_topic"]["0"] == 20
            assert data["filtering"]["papers_final"] == 90
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py::TestExportReportJson::test_exports_json_metadata -v`
Expected: NameError (export_report_json not defined)

**Step 3: Write minimal implementation**

Add to `src/pipeline/synthesis.py`:

```python
import json


def export_report_json(metrics: "PipelineMetrics", output_path: Path | str) -> None:
    """Export PipelineMetrics as structured JSON metadata.

    Parameters
    ----------
    metrics : PipelineMetrics
        Complete pipeline statistics.
    output_path : Path | str
        Path to save the .json file.
    """
    data = metrics.to_dict()

    # Convert dict keys to strings for JSON
    if "papers_per_topic" in data.get("topic_modeling", {}):
        topic_dict = data["topic_modeling"]["papers_per_topic"]
        data["topic_modeling"]["papers_per_topic"] = {
            str(k): v for k, v in topic_dict.items()
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py::TestExportReportJson::test_exports_json_metadata -v`
Expected: 1 passed

**Step 5: Update synthesis.py imports in test file**

Modify `tests/test_synthesis.py` imports:

```python
from pipeline.synthesis import (
    plot_topics,
    generate_report,
    _plot_horizontal_bar,
    plot_bibliometrics,
    export_report_tex,
    export_report_json,
)
```

**Step 6: Commit**

```bash
git add src/pipeline/synthesis.py tests/test_synthesis.py
git commit -m "feat(synthesis): add export_report_json for structured metadata

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update runner to collect and use PipelineMetrics

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `tests/test_runner.py`

**Step 1: Write the failing test**

Add to `tests/test_runner.py`:

```python
def test_exports_synthesis_metadata_json(config_with_tempdir, monkeypatch):
    """Verify runner exports synthesis_metadata.json."""
    config = config_with_tempdir

    # Mock user input for K selection
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    run_pipeline(config)

    json_path = config.processed_dir / "synthesis_metadata.json"
    assert json_path.exists()

    import json
    with open(json_path) as f:
        data = json.load(f)

    assert "ingestion" in data
    assert "topic_modeling" in data
    assert "filtering" in data
    assert "final_dataset" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_runner.py::test_exports_synthesis_metadata_json -v`
Expected: AssertionError (file doesn't exist)

**Step 3: Write minimal implementation**

Modify `src/pipeline/runner.py`:

```python
from pipeline.metrics import PipelineMetrics
from pipeline.synthesis import (
    generate_report,
    export_report_tex,
    export_report_json,
    plot_topics,
    plot_topic_audit,
    plot_bibliometrics,
)


def run_pipeline(config: PipelineConfig | None = None) -> None:
    """Execute the end-to-end SLR pipeline."""
    if config is None:
        config = PipelineConfig()

    logger.info("Starting pipeline run...")

    # ── Step 1 & 2: Ingest ─────────────────────────────────────────────
    logger.info(f"Ingesting raw data from {config.raw_dir}")
    df_raw, ingest_counts = ingest_all(config.raw_dir, config, return_counts=True)
    logger.info(f"Ingested {len(df_raw)} unique documents.")

    # ── Step 3: Preprocess ─────────────────────────────────────────────
    logger.info("Setting up NLTK resources...")
    setup_nltk()

    logger.info("Cleaning and tokenizing text...")
    text_data = (df_raw["title"] + " " + df_raw["abstract"]).tolist()
    tokens = [clean_text(doc) for doc in text_data]

    logger.info("Creating dictionary and corpus...")
    dictionary, corpus = create_corpus(tokens)
    logger.info(f"Dictionary size: {len(dictionary)} unique tokens.")

    # ── Step 4: Topic Modeling (Sweep) ─────────────────────────────────
    logger.info(f"Performing LDA sweep for K={config.topic_range}...")
    sweep_results = perform_lda_sweep(
        corpus=corpus,
        id2word=dictionary,
        texts=tokens,
        k_values=list(range(*config.topic_range)),
        passes=config.num_passes_sweep,
        random_state=config.random_state,
    )

    audit_plot_path = config.processed_dir / "topic_audit_plot.png"
    logger.info(f"Generating topic audit plot at {audit_plot_path}...")
    plot_topic_audit(sweep_results, audit_plot_path)

    candidates = select_top_candidates(sweep_results, n=3)
    valid_ks = {r.k for r in sweep_results}
    best_k = candidates[0].k

    print("\n══════════════════════════════════════════════════")
    print("Top K candidates (ranked by coherence):\n")
    print(f"  {'Rank':>4} │ {'K':>3} │ {'Coherence':>9} │ {'Log Perplexity':>14}")
    print(f"  {'─' * 4}─┼─{'─' * 3}─┼─{'─' * 9}─┼─{'─' * 14}")
    for i, c in enumerate(candidates, 1):
        print(f"  {i:>4} │ {c.k:>3} │ {c.coherence:>9.4f} │ {c.perplexity:>14.3f}")
    print(f"\n  Audit plot saved to: {audit_plot_path}")
    print("══════════════════════════════════════════════════\n")

    # Prompt user for K selection
    attempts = 0
    optimal_k = best_k
    while attempts < 3:
        raw = input(f"Select K [default={best_k}]: ").strip()
        if raw == "":
            optimal_k = best_k
            break
        try:
            chosen = int(raw)
            if chosen in valid_ks:
                optimal_k = chosen
                break
            else:
                print(f"  Invalid: K={chosen} not in sweep range. Valid: {sorted(valid_ks)}")
        except ValueError:
            print(f"  Invalid input. Enter a number.")
        attempts += 1
    else:
        logger.warning(f"Max attempts reached. Using best K={best_k}.")
        optimal_k = best_k

    selected_coherence = next(r.coherence for r in sweep_results if r.k == optimal_k)
    logger.info(f"Selected K={optimal_k} (Coherence: {selected_coherence:.4f})")

    # ── Step 5: Final Model ────────────────────────────────────────────
    logger.info(f"Training final model with K={optimal_k}...")
    final_model = train_final_model(
        corpus=corpus,
        id2word=dictionary,
        num_topics=optimal_k,
        passes=config.num_passes_final,
        random_state=config.random_state,
    )

    # ── Step 6: Identify & Label ───────────────────────────────────────
    logger.info("Generating topic labels...")
    labels = get_all_topic_labels(final_model)
    for tid, label in labels.items():
        logger.info(f"Topic {tid}: {label}")

    logger.info("Exporting topic terms with weights...")
    export_topic_terms(final_model, config.processed_dir)

    # ── Step 7: Selection ──────────────────────────────────────────────
    logger.info("Assigning dominant topics and filtering...")
    df_topics = assign_dominant_topic(df_raw, final_model, corpus)

    # Count papers per topic
    papers_per_topic = df_topics["Dominant_Topic"].value_counts().to_dict()

    df_selected, filter_stats = filter_documents(
        df_topics, config, full_df=df_topics, return_stats=True
    )
    logger.info(f"Selected {len(df_selected)} documents after filtering.")

    # ── Step 8: Quality Review Export ──────────────────────────────────
    review_path = config.processed_dir / "to_review.csv"
    logger.info(f"Exporting for manual review to {review_path}...")
    export_for_review(df_selected, review_path)

    # ── Step 9: Synthesis ──────────────────────────────────────────────
    logger.info("Generating synthesis reports...")

    # Build PipelineMetrics
    metrics = PipelineMetrics(
        scopus_raw=ingest_counts["scopus"],
        wos_raw=ingest_counts["wos"],
        duplicates_removed=ingest_counts["duplicates_removed"],
        unique_papers=ingest_counts["unique"],
        selected_k=optimal_k,
        coherence_score=selected_coherence,
        papers_per_topic=papers_per_topic,
        failed_topic_assignment=filter_stats["failed_topic_assignment"],
        passed_probability=filter_stats["passed_probability"],
        passed_citations=filter_stats["passed_citations"],
        papers_selected_strict=filter_stats["selected_strict"],
        papers_recovered=filter_stats["recovered"],
        papers_final=len(df_selected),
        year_min=int(df_selected["year"].min()) if len(df_selected) > 0 else 0,
        year_max=int(df_selected["year"].max()) if len(df_selected) > 0 else 0,
        total_citations=int(df_selected["cited_by"].sum()) if len(df_selected) > 0 else 0,
    )

    # Export reports
    tex_path = config.processed_dir / "synthesis_report.tex"
    logger.info(f"Exporting synthesis report to {tex_path}...")
    export_report_tex(metrics, tex_path)

    json_path = config.processed_dir / "synthesis_metadata.json"
    logger.info(f"Exporting synthesis metadata to {json_path}...")
    export_report_json(metrics, json_path)

    # Bibliometric plots
    logger.info("Generating bibliometric plots...")
    plot_bibliometrics(df_selected, config.processed_dir)

    # Topic plot
    plot_path = config.processed_dir / "topic_plot.html"
    logger.info(f"Generating topic map at {plot_path}...")
    plot_topics(final_model, corpus, dictionary, plot_path)

    logger.info("Pipeline run complete.")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_runner.py::test_exports_synthesis_metadata_json -v`
Expected: 1 passed

**Step 5: Run full runner test suite**

Run: `pytest tests/test_runner.py -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(runner): integrate PipelineMetrics and export synthesis_metadata.json

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Run full test suite and verify

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 128+ tests pass

**Step 2: Verify generated outputs**

If there's a test run with real data, check:
- `data/processed/synthesis_report.tex` has comprehensive table
- `data/processed/synthesis_metadata.json` exists and has all sections

**Step 3: Final commit (if needed)**

```bash
git add -A
git commit -m "docs: update synthesis report documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Completion Checklist

- [ ] PipelineMetrics dataclass created with to_dict() method
- [ ] ingest_all() returns source counts when requested
- [ ] filter_documents() returns filter stage stats when requested
- [ ] generate_report() accepts PipelineMetrics
- [ ] export_report_tex() formats comprehensive LaTeX table
- [ ] export_report_json() writes structured JSON metadata
- [ ] Runner collects metrics and exports both formats
- [ ] All tests pass (128+ tests)
- [ ] LaTeX table shows all pipeline stages
- [ ] JSON metadata has 4 top-level sections
