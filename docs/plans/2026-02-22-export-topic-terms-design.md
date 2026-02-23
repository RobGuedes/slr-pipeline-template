# Export Topic Terms with Weights

## Problem

After the pipeline runs, topic terms and their probabilities are logged to
console and then discarded. There is no persistent artifact for human
evaluation of topic quality.

## Solution

Add `export_topic_terms()` to `src/pipeline/topic_identify.py`. It extracts
the top N terms (default 10) with weights for every topic, generates a stub
label, and writes two files to `data/processed/`:

- **`topic_terms.csv`** — flat table: `topic_id, label, rank, term, weight`
- **`topic_terms.json`** — nested list of `{topic_id, label, terms: [{rank, term, weight}]}`

## Function Signature

```python
def export_topic_terms(
    model: LdaModel,
    out_dir: Path,
    topn: int = 10,
) -> pd.DataFrame:
```

Reuses existing `get_topic_terms()` and `generate_topic_label()` internally.
Returns the DataFrame for optional downstream use.

## Runner Integration

Called in `runner.py` after `get_all_topic_labels()` (around line 154),
saving to `config.processed_dir`.

## Output Examples

CSV:
```
topic_id,label,rank,term,weight
0,"risk, volatility, market",1,risk,0.0423
0,"risk, volatility, market",2,volatility,0.0312
```

JSON:
```json
[
  {
    "topic_id": 0,
    "label": "risk, volatility, market",
    "terms": [
      {"rank": 1, "term": "risk", "weight": 0.0423}
    ]
  }
]
```
