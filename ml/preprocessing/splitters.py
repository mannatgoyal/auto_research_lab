"""Deterministic train/validation/test splitting with cryptographic integrity verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import SplitConfig
from ml.datasets.loader import DatasetBundle, compute_dataframe_fingerprint


@dataclass(frozen=True)
class SplitData:
    """Immutable bundle containing partitioned datasets and partition integrity hashes."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    target_column: str
    feature_names: list[str]
    split_config: SplitConfig
    train_sha256: str
    val_sha256: str
    test_sha256: str

    @property
    def train_rows(self) -> int:
        return len(self.train_df)

    @property
    def val_rows(self) -> int:
        return len(self.val_df)

    @property
    def test_rows(self) -> int:
        return len(self.test_df)

    @property
    def total_rows(self) -> int:
        return self.train_rows + self.val_rows + self.test_rows


def create_splits(bundle: DatasetBundle, split_config: SplitConfig) -> SplitData:
    """Splits a dataset bundle into train, validation, and test sets deterministically.

    Guarantees:
    - Zero index overlap between train, val, and test sets.
    - Deterministic partition generation based strictly on split_config.seed.
    - Cryptographic SHA-256 fingerprint computed for each partition.
    """
    df = bundle.df.copy()
    target_col = bundle.target_column
    stratify = df[target_col] if split_config.strategy == "stratified" else None

    # Step 1: Separate test partition
    train_val_df, test_df = train_test_split(
        df,
        test_size=split_config.test_size,
        random_state=split_config.seed,
        stratify=stratify,
    )

    # Step 2: Separate validation partition from remaining data
    if split_config.val_size > 0:
        # Proportion of train_val_df that corresponds to val_size of total
        relative_val_size = split_config.val_size / (1.0 - split_config.test_size)
        stratify_val = (
            train_val_df[target_col] if split_config.strategy == "stratified" else None
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=split_config.seed,
            stratify=stratify_val,
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)

    # Sort indices to guarantee deterministic ordering
    train_df = train_df.sort_index()
    val_df = val_df.sort_index()
    test_df = test_df.sort_index()

    # Integrity verification: zero overlap
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    if train_idx.intersection(val_idx) or train_idx.intersection(test_idx) or val_idx.intersection(test_idx):
        raise RuntimeError("Data splitting integrity violation: detected overlapping indices across partitions!")

    return SplitData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=target_col,
        feature_names=bundle.feature_names,
        split_config=split_config,
        train_sha256=compute_dataframe_fingerprint(train_df),
        val_sha256=compute_dataframe_fingerprint(val_df) if len(val_df) > 0 else "empty",
        test_sha256=compute_dataframe_fingerprint(test_df),
    )
