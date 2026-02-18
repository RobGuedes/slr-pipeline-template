"""Step 8 — Quality Assessment: manual review support.

Functions to export the filtered dataset for manual inspection and to load back
the user's decisions (Keep/Reject).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_for_review(df: pd.DataFrame, path: Path) -> None:
    """Save the dataframe to CSV with an empty 'Keep' column for manual review.

    The user should fill the 'Keep' column with 'y', 'yes', '1', or 'true'
    to retain the paper.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataset (from Step 7).
    path : Path
        Destination for the CSV file.
    """
    df_out = df.copy()
    if "Keep" not in df_out.columns:
        df_out["Keep"] = ""
    
    # Ensure parent dir exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False, encoding="utf-8-sig")


def load_reviewed(path: Path) -> pd.DataFrame:
    """Load the reviewed CSV and filter for papers marked as 'Keep'.

    Accepts 'y', 'yes', '1', 'true' (case-insensitive) as confirmation.

    Parameters
    ----------
    path : Path
        Path to the reviewed CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the kept papers.
    
    Raises
    ------
    ValueError
        If the 'Keep' column is missing.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    if "Keep" not in df.columns:
        raise ValueError(f"Column 'Keep' not found in {path}. "
                         "Please add a 'Keep' column to mark papers.")
    
    # Normalize 'Keep' to lowercase string
    keep_col = df["Keep"].astype(str).str.lower().str.strip()
    
    # Filter
    valid_keep = {"y", "yes", "1", "true"}
    mask = keep_col.isin(valid_keep)
    
    return df[mask].reset_index(drop=True)
