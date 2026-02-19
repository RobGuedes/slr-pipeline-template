# Design: Interactive K Selection

**Date:** 2026-02-19
**Status:** Approved
**Branch:** claude/stoic-ptolemy

## Problem

The pipeline automatically picks the single highest-coherence K and trains
one final model. The user has no opportunity to inspect candidates or choose
a K that better fits their research context (e.g., preferring fewer broader
topics vs. more granular ones).

## Decision

**Approach A — Interactive `input()` prompt.**

After the LDA sweep and audit plot, the pipeline:

1. Sorts sweep results by coherence descending, takes top 3.
2. Prints a formatted table showing K, coherence, and log perplexity.
3. Points the user to the saved audit plot for visual inspection.
4. Prompts via `input()` for the user's chosen K (default = best).
5. Validates input — any K from the sweep range is accepted, not just top 3.
6. Re-prompts up to 3 times on invalid input, then falls back to best K.
7. Trains one final model with the selected K.

## Changes

### `topic_model.py`

Add a pure helper:

```python
def select_top_candidates(
    sweep_results: list[SweepResult], n: int = 3
) -> list[SweepResult]:
```

Returns top N results sorted by coherence descending. Testable in isolation.

### `runner.py`

Replace:

```python
best_result = max(sweep_results, key=lambda x: x.coherence)
optimal_k = best_result.k
```

With:

1. Call `select_top_candidates(sweep_results, n=3)`.
2. Print candidate table to stdout.
3. Log path to audit plot.
4. `input("Select K [default=<best>]: ")`.
5. Validate against all swept K values.
6. Assign `optimal_k` from user choice.

### No changes to

- `config.py` — no new config fields
- `preprocess.py` — untouched
- `topic_model.py` — sweep logic untouched, only new helper added
- `synthesis.py` — untouched
- All downstream steps — untouched

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Top 3 by coherence (simple sort) | Predictable, no false positives from noisy curves |
| 2 | Fixed N=3 candidates | Matches user preference, avoids config bloat |
| 3 | Accept any swept K, not just top 3 | User may have domain reasons to pick a non-peak K |
| 4 | Re-prompt up to 3 times | Graceful handling of typos without infinite loops |
| 5 | Fall back to best K after 3 failures | Pipeline doesn't crash on bad input |
| 6 | No config override field | YAGNI — user said Approach A only |
