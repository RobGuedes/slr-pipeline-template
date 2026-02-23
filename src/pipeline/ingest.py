"""Step 3 — Literature Search: load, merge, de-duplicate.

Replaces the 700+ lines of ``paperUtils.openFileToDict`` and
``removeDuplicates`` with idiomatic pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from unidecode import unidecode

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig

# ── Canonical column names ────────────────────────────────────────────
# Every loader must produce a DataFrame with at least these columns.
_CANONICAL = (
    "author",
    "title",
    "year",
    "doi",
    "abstract",
    "cited_by",
    "document_type",
    "author_keywords",
    "source_title",
    "source_db",
    "affiliations",
)

# ── Scopus → canonical mapping ────────────────────────────────────────
_SCOPUS_MAP = {
    "Authors": "author",
    "Title": "title",
    "Year": "year",
    "DOI": "doi",
    "Abstract": "abstract",
    "Cited by": "cited_by",
    "Document Type": "document_type",
    "Author Keywords": "author_keywords",
    "Source title": "source_title",
    "Affiliations": "affiliations",
}

# ── WoS → canonical mapping ──────────────────────────────────────────
_WOS_MAP = {
    "AU": "author",
    "TI": "title",
    "PY": "year",
    "DI": "doi",
    "AB": "abstract",
    "TC": "cited_by",
    "DT": "document_type",
    "DE": "author_keywords",
    "SO": "source_title",
    "C1": "affiliations",
}


# ── Loaders ───────────────────────────────────────────────────────────


def load_scopus_csv(path: Path) -> pd.DataFrame:
    """Read a Scopus CSV export and normalise columns.

    Parameters
    ----------
    path : Path
        Path to a ``.csv`` file exported from Scopus.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names and ``source_db="Scopus"``.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=_SCOPUS_MAP)
    df["source_db"] = "Scopus"
    # Ensure canonical columns exist, even if empty
    for col in _CANONICAL:
        if col not in df.columns:
            df[col] = ""
    df["cited_by"] = pd.to_numeric(df["cited_by"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df


def load_wos_txt(path: Path) -> pd.DataFrame:
    """Read a WoS tab-delimited TXT export and normalise columns.

    Parameters
    ----------
    path : Path
        Path to a ``.txt`` file exported from Web of Science.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names and ``source_db="WoS"``.
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df = df.rename(columns=_WOS_MAP)
    df["source_db"] = "WoS"
    for col in _CANONICAL:
        if col not in df.columns:
            df[col] = ""
    df["cited_by"] = pd.to_numeric(df["cited_by"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df


# ── Merge ─────────────────────────────────────────────────────────────


def merge_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Vertically concatenate DataFrames from different sources.

    Resets the index so there are no duplicate row labels.
    """
    return pd.concat(dfs, ignore_index=True)


# ── De-duplication ────────────────────────────────────────────────────


def _normalise_title(title: str) -> str:
    """Strip accents, uppercase, remove non-alphanumeric chars."""
    if not isinstance(title, str):
        return ""
    t = unidecode(title).upper().strip()
    return "".join(c for c in t if c.isalnum())


def _first_author_last_name(author: str) -> str:
    """Extract the first author's last name (normalised)."""
    if not isinstance(author, str):
        return ""
    a = unidecode(author).upper().strip()
    # Split on common delimiters: ";" (Scopus), "," (name parts)
    first = a.split(";")[0].split(",")[0].strip()
    return "".join(c for c in first if c.isalpha())


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate papers.

    Two papers are considered duplicates if:
    1. They share the same non-empty DOI, **or**
    2. Their normalised title AND first-author last name match.

    When duplicates are found, ``cited_by`` is averaged across copies.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with canonical columns.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame.
    """
    df = df.copy()
    df["_norm_title"] = df["title"].apply(_normalise_title)
    df["_first_author"] = df["author"].apply(_first_author_last_name)

    # Phase 1: DOI-based dedup — group by non-empty DOI
    doi_mask = df["doi"].notna() & (df["doi"] != "")
    doi_groups = df[doi_mask].groupby("doi", sort=False)

    # Collect indices to drop (keep first, average citations)
    keep_indices: list[int] = []
    drop_indices: list[int] = []
    avg_citations: dict[int, int] = {}

    for _, group in doi_groups:
        if len(group) > 1:
            keep_idx = group.index[0]
            keep_indices.append(keep_idx)
            drop_indices.extend(group.index[1:].tolist())
            avg_citations[keep_idx] = int(group["cited_by"].mean())
        else:
            keep_indices.append(group.index[0])

    # Apply DOI dedup
    df = df.drop(index=drop_indices)
    for idx, avg in avg_citations.items():
        df.loc[idx, "cited_by"] = avg

    # Phase 2: Title+Author dedup (for papers without DOI or different DOIs)
    df = df.sort_values(by=["_norm_title", "_first_author"]).reset_index(drop=True)
    title_author_mask = df.duplicated(subset=["_norm_title", "_first_author"], keep="first")
    df = df[~title_author_mask].reset_index(drop=True)

    # Clean up helper columns
    df = df.drop(columns=["_norm_title", "_first_author"])
    return df


# ── Orchestrator ──────────────────────────────────────────────────────


def ingest_all(
    raw_dir: Path,
    config: "PipelineConfig",
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """End-to-end ingestion: load every file in *raw_dir*, merge, filter, dedup.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``.csv`` (Scopus) and ``.txt`` (WoS) files.
    config : PipelineConfig
        Pipeline configuration (used for ``included_doc_types``).
    return_counts : bool, optional
        If True, return a tuple of (DataFrame, counts_dict).
        counts_dict contains: scopus, wos, unique, duplicates_removed.
        Default is False.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict[str, int]]
        Clean, de-duplicated DataFrame ready for preprocessing.
        If return_counts=True, returns (DataFrame, counts_dict).
    """
    frames: list[pd.DataFrame] = []
    raw = Path(raw_dir)
    scopus_count = 0
    wos_count = 0

    for f in sorted(raw.iterdir()):
        if f.suffix == ".csv":
            df = load_scopus_csv(f)
            frames.append(df)
            scopus_count += len(df)
        elif f.suffix == ".txt":
            df = load_wos_txt(f)
            frames.append(df)
            wos_count += len(df)

    if not frames:
        msg = f"No CSV or TXT files found in {raw_dir}"
        raise FileNotFoundError(msg)

    merged = merge_sources(*frames)

    # Filter by document type
    merged = merged[merged["document_type"].isin(config.included_doc_types)]
    pre_dedup_count = len(merged)

    # De-duplicate
    result = deduplicate(merged)
    result = result.reset_index(drop=True)
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
