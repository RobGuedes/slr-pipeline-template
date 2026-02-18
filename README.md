# SLR Pipeline

An automated **Systematic Literature Review (SLR)** pipeline built on top of ScientoPy.

## Structure

```
slr-pipeline/
├── .claude/                  # Agent instructions and skills
│   ├── CLAUDE.md
│   └── skills/
├── legacy/                   # Read-only reference material
│   ├── scientopy/
│   ├── notebooks/
│   └── methodology.md
├── src/
│   └── pipeline/             # Main pipeline code (agents build this)
├── tests/                    # Test suite
├── data/
│   ├── raw/                  # Raw Scopus/WoS exports
│   └── processed/            # Pipeline outputs
└── pyproject.toml
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Methodology

See [`legacy/methodology.md`](legacy/methodology.md) for the full 9-step SLR methodology.
