"""Data Loader Module - Loading and sanitizing material data from CSV"""

import pandas as pd
import numpy as np
import os
from typing import Optional


def load_data(csv_path: str, id_column: str, required_cols: list = None) -> pd.DataFrame:
    """Load and sanitize material data from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}")

    original_size = len(df)

    if required_cols is None:
        required_cols = [id_column]
    else:
        if id_column not in required_cols:
            required_cols = [id_column] + required_cols

    df_clean = df.dropna(subset=required_cols)

    rows_dropped = original_size - len(df_clean)
    if rows_dropped > 0:
        print(f"[Data Loader] Dropped {rows_dropped} rows with missing values ({original_size} → {len(df_clean)})")
    else:
        print(f"[Data Loader] All {len(df_clean)} rows have valid data")

    if df_clean.empty:
        raise ValueError("No valid data remaining after cleaning")

    return df_clean.reset_index(drop=True)


def validate_required_columns(df: pd.DataFrame, required_cols: list,
                             operation: str = "this operation") -> None:
    """Validate that required columns exist in DataFrame"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for {operation}: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )