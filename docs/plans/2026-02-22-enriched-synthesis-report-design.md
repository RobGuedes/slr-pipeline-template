# Enriched Synthesis Report

## Problem

The current synthesis report (`synthesis_report.tex`) contains only 3 metrics: total papers, year range, and total citations. It doesn't track pipeline attrition (papers filtered at each stage), source-level statistics, topic distribution, or recency recovery effectiveness.

For methodology transparency and reproducibility, researchers need comprehensive metrics showing the full data flow through the pipeline.

## Solution

Create a `PipelineMetrics` dataclass to accumulate statistics at each pipeline stage. Thread this through the runner and generate two outputs:

- **`synthesis_report.tex`** — LaTeX table (human-readable, for paper methodology sections)
- **`synthesis_metadata.json`** — Structured JSON (machine-parseable, for reproducibility/analysis)

## Metrics to Capture

### 1. Ingestion Stage
- Papers from Scopus (raw count before dedup)
- Papers from WoS (raw count before dedup)
- Total unique papers after deduplication
- Duplicates removed

### 2. Topic Modeling Stage
- Selected K (number of topics)
- Coherence score of the selected model
- Papers per topic (distribution across the K topics)

### 3. Filtering Stage
- Papers with failed topic assignment (excluded)
- Papers passing probability threshold
- Papers passing citation threshold
- Total papers selected (after strict filters)

### 4. Recency Recovery Stage
- Papers recovered via recency criteria
- Final selected papers (strict + recovered)

### 5. Final Dataset Stats
- Year range (min and max)
- Total citations

## Data Structure

```python
@dataclass
class PipelineMetrics:
    """Comprehensive pipeline statistics for synthesis reporting."""

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
```

## Data Flow

The runner accumulates metrics as the pipeline executes:

1. **After `ingest_all()`** → Count papers per source (`source_db` column), compute unique count and duplicates removed
2. **After `train_final_model()`** → Store selected K and coherence score
3. **After `assign_dominant_topic()`** → Count papers per topic from `Dominant_Topic` column
4. **During/after `filter_documents()`** → Track each filter stage (topic assignment failures, probability threshold, citation threshold, strict selection, recovery)
5. **After filtering** → Compute final year range and total citations

## Modified `filter_documents()` Signature

To expose recovery stats, `filter_documents()` will optionally return a tuple:

```python
def filter_documents(
    df: pd.DataFrame,
    config: PipelineConfig,
    full_df: pd.DataFrame | None = None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """
    ...

    Returns
    -------
    pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]
        Filtered DataFrame. If return_stats=True, returns (df, stats)
        where stats = {"strict": int, "recovered": int}.
    """
```

## Output Formats

### LaTeX Table (synthesis_report.tex)

Organized into sections matching pipeline stages:

```latex
\begin{table}[h]
\centering
\begin{tabular}{lr}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline
\multicolumn{2}{c}{\textit{Ingestion}} \\
Scopus (raw) & 450 \\
WoS (raw) & 320 \\
Duplicates removed & 85 \\
Unique papers & 685 \\
\hline
\multicolumn{2}{c}{\textit{Topic Modeling}} \\
Topics (K) & 8 \\
Coherence & 0.542 \\
\hline
\multicolumn{2}{c}{\textit{Filtering}} \\
Failed topic assignment & 12 \\
Passed probability (≥0.7) & 520 \\
Passed citations & 420 \\
Selected (strict) & 380 \\
Recovered (recency) & 35 \\
\textbf{Final selected} & \textbf{415} \\
\hline
\multicolumn{2}{c}{\textit{Final Dataset}} \\
Year range & 2016 -- 2026 \\
Total citations & 7,254 \\
\hline
\end{tabular}
\caption{Synthesis Report Statistics}
\label{tab:synthesis_report}
\end{table}
```

### JSON Metadata (synthesis_metadata.json)

Full structured export with per-topic distribution:

```json
{
  "ingestion": {
    "scopus_raw": 450,
    "wos_raw": 320,
    "duplicates_removed": 85,
    "unique_papers": 685
  },
  "topic_modeling": {
    "selected_k": 8,
    "coherence_score": 0.542,
    "papers_per_topic": {
      "0": 85,
      "1": 72,
      ...
    }
  },
  "filtering": {
    "failed_topic_assignment": 12,
    "passed_probability": 520,
    "passed_citations": 420,
    "papers_selected_strict": 380,
    "papers_recovered": 35,
    "papers_final": 415
  },
  "final_dataset": {
    "year_min": 2016,
    "year_max": 2026,
    "total_citations": 7254
  }
}
```

## Implementation Notes

- `PipelineMetrics` will live in a new `src/pipeline/metrics.py` module
- Enhanced `generate_report()` will accept a `PipelineMetrics` instance
- New `export_report_json()` function will write the JSON metadata
- Runner will create and populate `PipelineMetrics` incrementally
- Backward compatibility: existing `generate_report(df)` signature still works for simple stats
