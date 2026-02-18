# SLR Pipeline Refactoring

## Objective
Refactor legacy ScientoPy code and Jupyter notebooks into a single, clean,
tested Python pipeline that executes a systematic literature review methodology.

## Context
The methodology has 9 steps. Steps 1–2 are manual (human decisions).
Steps 3–9 are partially or fully automatable. This project builds the
automation pipeline for steps 3–9.

## Source Materials (read-only — do NOT modify)
- `legacy/scientopy/` — ScientoPy v2.1.5 (Python 3.6, no tests, global state)
- `legacy/notebooks/` — Two Jupyter notebooks with NLP preprocessing and topic modeling
- `legacy/methodology.md` — The 9-step methodology these tools must implement

## The Pipeline Steps (map to methodology)

The final pipeline must expose these steps as callable modules:

| Step | Name | Input | Output |
|------|------|-------|--------|
| 3 | `merge_and_deduplicate` | Raw Scopus CSV + WoS exports | Unified deduplicated DataFrame |
| 4 | `preprocess_text` | Deduplicated DataFrame | Tokenized, lemmatized, TF-IDF matrix |
| 5 | `fit_topic_models` | TF-IDF matrix + corpus | LDA models with coherence/perplexity scores |
| 6 | `identify_topics` | Best LDA model | Top terms per topic + proposed labels |
| 7 | `select_topics` | Topics + document-topic matrix + citation data | Filtered documents (probability > threshold, citations > threshold) |
| 8 | `assess_quality` | Selected documents | Flagged abstracts for human review |
| 9 | `synthesize` | Full dataset + selected documents | Bibliometric statistics + per-topic summaries |

Each step is a Python module in `src/pipeline/`. Each receives the output of
the previous step. Data flows as DataFrames or named tuples — no global state.

## Engineering Rules
- Python 3.13, type hints on all public functions
- Google-style docstrings
- Black (88 chars), Ruff linting
- TDD mandatory — use the test-driven-development skill
- No global mutable state (eliminate ScientoPy's globalVar pattern)
- Each module must be independently testable
- Pandas DataFrames as the interchange format between steps
- All stochastic operations (LDA) must accept a `random_state` parameter

## What to Extract from Legacy Code
- `legacy/scientopy/`: merge logic, deduplication, bibliometric analysis, graph generation
- `legacy/notebooks/`: NLP preprocessing, LDA fitting, coherence optimization, topic extraction

## What NOT to Do
- Do not preserve ScientoPy's CLI interface or argparse — this is a library, not a CLI tool
- Do not preserve the GUI code
- Do not write methodology documentation — that's a separate phase
- Do not create triple-layer docs — that's a separate phase

## Dependencies
Core: numpy, pandas, scipy, scikit-learn, gensim, nltk, matplotlib, seaborn, pyLDAvis
Optional: litstudy, wordcloud, unidecode

## Mandatory: Superpowers Skills

Before doing ANY work in this project, you MUST read and follow the skills
in `.claude/skills/`. Start every task by reading
`.claude/skills/using-superpowers/SKILL.md` and following its decision flow.

The skill chain for feature work is:

1. **brainstorming** → Explore the idea, ask questions, propose approaches, get design approval
2. **writing-plans** → Create a detailed implementation plan from the approved design
3. **executing-plans** → Execute the plan step-by-step with checkpoints
4. **test-driven-development** → RED-GREEN-REFACTOR for every module (rigid — follow exactly)

For debugging: **systematic-debugging** → **verification-before-completion**

This is non-negotiable. Do not skip skills to "save time."
If there is even a 1% chance a skill applies, you MUST invoke it.