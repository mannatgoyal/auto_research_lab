"""Deterministic dataset loading and cryptographic fingerprinting."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml.config import DatasetConfig


@dataclass(frozen=True)
class DatasetBundle:
    """Immutable bundle containing a loaded dataset and its cryptographic metadata."""
    df: pd.DataFrame
    feature_names: List[str]
    target_column: str
    source: str
    fingerprint_sha256: str

    @property
    def num_rows(self) -> int:
        return len(self.df)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def compute_dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Computes a deterministic SHA-256 hash of a DataFrame's data, schema, and index."""
    hasher = hashlib.sha256()
    # Hash column names and dtypes
    schema_repr = "|".join(f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes))
    hasher.update(schema_repr.encode("utf-8"))
    # Hash values deterministically
    hasher.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return hasher.hexdigest()


def load_dataset(config: DatasetConfig) -> DatasetBundle:
    """Loads a tabular dataset based on DatasetConfig, returning a validated DatasetBundle.

    Supported standard datasets:
    - "breast_cancer": UCI Breast Cancer Wisconsin (Diagnostic). 569 instances, 30 features,
      binary classification (0=malignant, 1=benign). Offline and deterministic.
    - Path to local CSV or Parquet file.
    """
    dataset_name = config.name.strip().lower()

    if dataset_name in ("breast_cancer", "wisconsin_breast_cancer"):
        raw = load_breast_cancer(as_frame=True)
        df = raw.frame.copy()
        target_col = "target"
        # Ensure target column is integer binary
        df[target_col] = df[target_col].astype(int)
        feature_names = [col for col in df.columns if col != target_col]
        source = "UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)"

    elif os.path.exists(config.name):
        filepath = config.name
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".parquet") or filepath.endswith(".pq"):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format for dataset: {filepath}")

        if config.target_column is None:
            raise ValueError(f"target_column must be specified when loading file: {filepath}")
        if config.target_column not in df.columns:
            raise ValueError(
                f"Target column '{config.target_column}' not found in columns: {list(df.columns)}"
            )

        target_col = config.target_column
        feature_names = [col for col in df.columns if col != target_col]
        source = f"Local File: {os.path.abspath(filepath)}"

    else:
        raise ValueError(
            f"Unknown or missing dataset '{config.name}'. "
            f"Supported built-in: 'breast_cancer' or specify a valid local file path."
        )

    # Optional target override if specified and exists
    if config.target_column and config.target_column in df.columns:
        target_col = config.target_column
        feature_names = [col for col in df.columns if col != target_col]

    fingerprint = compute_dataframe_fingerprint(df)

    return DatasetBundle(
        df=df,
        feature_names=feature_names,
        target_column=target_col,
        source=config.source or source,
        fingerprint_sha256=fingerprint,
    )
