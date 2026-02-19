# Interactive K Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace automatic max-coherence K selection with an interactive prompt that shows top 3 candidates and lets the user choose.

**Architecture:** Add a pure `select_top_candidates()` helper in `topic_model.py` (testable), then rewire `runner.py` to print candidates and call `input()`. No new modules, no config changes.

**Tech Stack:** Python 3.13, pytest, unittest.mock

---

### Task 1: Add `select_top_candidates()` helper

**Files:**
- Modify: `src/pipeline/topic_model.py` (add function after `SweepResult`)
- Test: `tests/test_topic_model.py` (add new test class)

**Step 1: Write the failing test**

Add to `tests/test_topic_model.py`:

```python
from pipeline.topic_model import select_top_candidates

class TestSelectTopCandidates:
    def test_returns_top_n_by_coherence(self):
        """Top 3 from 5 results, sorted by coherence descending."""
        results = [
            SweepResult(k=2, coherence=0.30, perplexity=-7.0),
            SweepResult(k=3, coherence=0.50, perplexity=-6.5),
            SweepResult(k=4, coherence=0.45, perplexity=-6.8),
            SweepResult(k=5, coherence=0.60, perplexity=-6.2),
            SweepResult(k=6, coherence=0.40, perplexity=-6.9),
        ]
        top = select_top_candidates(results, n=3)
        assert len(top) == 3
        assert top[0].k == 5   # highest coherence
        assert top[1].k == 3   # second
        assert top[2].k == 4   # third

    def test_returns_all_if_fewer_than_n(self):
        """If only 2 results exist, return both (don't crash)."""
        results = [
            SweepResult(k=2, coherence=0.30, perplexity=-7.0),
            SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        ]
        top = select_top_candidates(results, n=3)
        assert len(top) == 2
        assert top[0].k == 3
        assert top[1].k == 2

    def test_default_n_is_3(self):
        """Verify default n=3 without passing argument."""
        results = [
            SweepResult(k=i, coherence=i * 0.1, perplexity=-7.0)
            for i in range(2, 8)
        ]
        top = select_top_candidates(results)
        assert len(top) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic_model.py::TestSelectTopCandidates -v`
Expected: FAIL with `ImportError: cannot import name 'select_top_candidates'`

**Step 3: Write minimal implementation**

Add to `src/pipeline/topic_model.py` after `SweepResult`:

```python
def select_top_candidates(
    sweep_results: list[SweepResult], n: int = 3
) -> list[SweepResult]:
    """Return the top N sweep results ranked by coherence descending.

    Args:
        sweep_results: Full list of sweep results from perform_lda_sweep.
        n: Number of candidates to return.

    Returns:
        Top N results sorted by coherence (highest first).
    """
    return sorted(sweep_results, key=lambda r: r.coherence, reverse=True)[:n]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic_model.py::TestSelectTopCandidates -v`
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/pipeline/topic_model.py tests/test_topic_model.py
git commit -m "feat(topic_model): add select_top_candidates helper"
```

---

### Task 2: Add interactive prompt to runner

**Files:**
- Modify: `src/pipeline/runner.py:93-96` (replace K selection logic)
- Modify: `src/pipeline/runner.py:20` (add import for `select_top_candidates`)
- Test: `tests/test_runner.py` (add new tests)

**Step 1: Write the failing tests**

Add to `tests/test_runner.py`:

```python
@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.convert_to_litstudy")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", return_value="3")
def test_interactive_k_selection(
    mock_input, mock_labels, mock_export_review, mock_plot_biblio,
    mock_plot_audit, mock_plot, mock_tex, mock_report, mock_litstudy,
    mock_filter, mock_assign, mock_train, mock_top_candidates, mock_sweep,
    mock_corpus, mock_clean, mock_ingest
):
    """Pipeline prompts user and trains model with chosen K."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame({
        "title": ["Paper A"], "abstract": ["Abstract A"]
    })
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    all_results = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_sweep.return_value = all_results

    # Top candidates returns k=5 first (highest coherence)
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    # User typed "3", so final model should use K=3
    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 3

    # input() was called
    mock_input.assert_called_once()


@patch("pipeline.runner.ingest_all")
@patch("pipeline.runner.clean_text")
@patch("pipeline.runner.create_corpus")
@patch("pipeline.runner.perform_lda_sweep")
@patch("pipeline.runner.select_top_candidates")
@patch("pipeline.runner.train_final_model")
@patch("pipeline.runner.assign_dominant_topic")
@patch("pipeline.runner.filter_documents")
@patch("pipeline.runner.convert_to_litstudy")
@patch("pipeline.runner.generate_report")
@patch("pipeline.runner.export_report_tex")
@patch("pipeline.runner.plot_topics")
@patch("pipeline.runner.plot_topic_audit")
@patch("pipeline.runner.plot_bibliometrics")
@patch("pipeline.runner.export_for_review")
@patch("pipeline.runner.get_all_topic_labels")
@patch("builtins.input", return_value="")
def test_empty_input_uses_default_best_k(
    mock_input, mock_labels, mock_export_review, mock_plot_biblio,
    mock_plot_audit, mock_plot, mock_tex, mock_report, mock_litstudy,
    mock_filter, mock_assign, mock_train, mock_top_candidates, mock_sweep,
    mock_corpus, mock_clean, mock_ingest
):
    """Empty input (just Enter) should use the default best K."""
    from pipeline.topic_model import SweepResult

    mock_ingest.return_value = pd.DataFrame({
        "title": ["Paper A"], "abstract": ["Abstract A"]
    })
    mock_corpus.return_value = (MagicMock(name="dict"), MagicMock(name="corpus"))

    mock_sweep.return_value = [
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
    ]
    mock_top_candidates.return_value = [
        SweepResult(k=5, coherence=0.60, perplexity=-6.2),
        SweepResult(k=3, coherence=0.50, perplexity=-6.5),
    ]

    mock_train.return_value = MagicMock(name="lda_model")

    cfg = PipelineConfig(raw_dir=Path("/tmp/raw"), processed_dir=Path("/tmp/proc"))
    run_pipeline(cfg)

    # Default = best K = 5
    train_kwargs = mock_train.call_args.kwargs
    assert train_kwargs["num_topics"] == 5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py::test_interactive_k_selection tests/test_runner.py::test_empty_input_uses_default_best_k -v`
Expected: FAIL (runner doesn't import `select_top_candidates` or call `input()`)

**Step 3: Write the implementation**

In `src/pipeline/runner.py`, update the import line:

```python
from pipeline.topic_model import perform_lda_sweep, train_final_model, select_top_candidates
```

Replace lines 93-96 (the current K selection) with:

```python
    # Identify top candidate Ks
    candidates = select_top_candidates(sweep_results, n=3)
    valid_ks = {r.k for r in sweep_results}
    best_k = candidates[0].k

    # Present candidates to user
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

    logger.info(f"Selected K={optimal_k} (Coherence: {next(r.coherence for r in sweep_results if r.k == optimal_k):.4f})")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: ALL PASSED

**Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASSED (81+ tests)

**Step 6: Commit**

```bash
git add src/pipeline/runner.py tests/test_runner.py
git commit -m "feat(runner): interactive K selection with top-3 candidates"
```

---

### Task 3: Update existing `test_end_to_end_flow`

The existing test in `test_runner.py` will break because `runner.py` now
imports `select_top_candidates` and calls `input()`. It needs patching.

**Files:**
- Modify: `tests/test_runner.py` (update existing test)

**Step 1: Update the existing test**

Add these patches to `test_end_to_end_flow`:
- `@patch("pipeline.runner.select_top_candidates")`
- `@patch("builtins.input", return_value="")`

Update the mock setup so `select_top_candidates` returns the same mock
sweep result, and `input` returns empty string (default K).

**Step 2: Run full suite to verify**

Run: `pytest tests/ -v`
Expected: ALL PASSED

**Step 3: Commit**

```bash
git add tests/test_runner.py
git commit -m "fix(tests): patch select_top_candidates and input in e2e test"
```

---

### Execution Notes

- Tasks 2 and 3 are closely related — Task 3 fixes the existing test that
  Task 2 breaks. They can be done in a single commit if preferred.
- The `input()` call is mockable via `@patch("builtins.input")`.
- `print()` output is not tested (cosmetic) — only the K selection logic
  and `train_final_model` call are verified.
