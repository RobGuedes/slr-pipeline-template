# Recency-Aware Citation Filter & Unified Plot Identity

**Date:** 2026-02-20
**Status:** Approved

## Problem

The global `min_citations=10` default creates recency bias: papers published
in the last few years cannot accumulate 10 citations regardless of quality.
This excludes potentially groundbreaking recent work from the SLR results.

Additionally, bibliometric plots use inconsistent styling (mix of litstudy
defaults and custom matplotlib), and there is no "top publication sources"
plot.

## Design Decisions

### Recency-Aware Citation Recovery (Two-Pass Approach)

**Approach:** Keep the existing strict citation filter as pass 1. Add a
second pass (`recover_recent_papers`) that scans rejected papers and adds
back those meeting age-based criteria. Final result is the union of both
sets.

**Age brackets (year-level granularity):**

| Age (years)       | Rule                                                      |
|-------------------|-----------------------------------------------------------|
| < 2               | 0 citations OK if author is top-15 OR source is top-15   |
| 2 to < 6          | At least `mid_range_min_citations` (default 5)            |
| >= 6              | Original `min_citations` threshold (default 10)           |

- Age = `reference_year - publication_year`
- `reference_year` defaults to current year if not set in config
- Top-15 authors and top-15 sources computed from the **full dataset**
  (before any filtering)
- Author matching: a paper qualifies if **any** of its authors (semicolon-
  separated) is in the top-15 list
- Source matching: direct match on `Source title` column

### Configuration (new fields in `TopicSelectConfig`)

```python
recency_filter_enabled: bool = True
recent_threshold_years: int = 2
mid_range_threshold_years: int = 6
mid_range_min_citations: int = 5
top_n_authors: int = 15
top_n_sources: int = 15
reference_year: int | None = None  # defaults to current year
```

Existing fields (`min_citations`, `min_citations_per_topic`, `min_topic_prob`)
remain unchanged.

### Unified Plot Identity

**Remove litstudy from synthesis.py:** Replace all litstudy plot calls with
custom matplotlib plots. Remove `convert_to_litstudy()`, `litstudy` import,
and `DocumentSet` parameter from `plot_bibliometrics()`. litstudy remains
installed as a project dependency but is no longer used for plotting.

**Shared helper:** `_plot_horizontal_bar(data, xlabel, title, output_path)`
encapsulates common bar chart styling.

**Visual identity for all bar charts:**
- Color: `#4c72b0`
- Figsize: (10, 6)
- DPI: 150
- `tight_layout()`
- Horizontal bars with `invert_yaxis()` (descending order)
- X-axis label: "No. of documents"

**Publication years plot:** Custom vertical bar chart (same color/DPI/layout,
but vertical bars — distinct from the horizontal bar charts).

**Plot inventory after redesign:**

| Plot                | Style            | Source             |
|---------------------|------------------|--------------------|
| Publication years   | Vertical bars    | Custom (replacing litstudy) |
| Top authors         | Horizontal bars  | Custom (replacing litstudy) |
| Top sources         | Horizontal bars  | **New**            |
| Top countries       | Horizontal bars  | Refactored to helper |
| Top affiliations    | Horizontal bars  | Refactored to helper |

### Code Cleanup

- Remove unused `present_topics` variable in `topic_select.py` (line 113)
- Clarify the `-1` topic filter comment (line 116)
- Remove `convert_to_litstudy` import from `runner.py`

### Data Flow

```
full_df (all papers)
  │
  ├─ compute top_15_authors (by paper count)
  ├─ compute top_15_sources (by paper count)
  │
  ▼
filter_documents(df, config)
  │
  ├─ Pass 1: strict filter (prob + citations)
  │    → accepted papers
  │    → rejected papers
  │
  ├─ Pass 2: recover_recent_papers(rejected, config, top_authors, top_sources)
  │    → recovered papers
  │
  └─ Final: accepted ∪ recovered
```

### Testing Plan

- Unit tests for `recover_recent_papers()` (all 3 age brackets)
- Unit tests for top-15 author/source computation
- Unit tests for edge cases (missing year, missing source title, no authors)
- Unit tests for each plot function (PNG output, non-empty)
- Integration test: two-pass filter produces correct union
- Regression: existing `filter_documents` tests still pass
