"""Deterministic dataset profiling and exploratory statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ml.datasets.loader import DatasetBundle


@dataclass
class FeatureStat:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_numeric: bool
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None


@dataclass
class DatasetProfile:
    dataset_name: str
    row_count: int
    column_count: int
    target_column: str
    features: List[FeatureStat]
    target_counts: Dict[str, int]
    target_proportions: Dict[str, float]
    imbalance_ratio: float
    high_correlation_pairs: List[Dict[str, Any]]
    fingerprint_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def profile_dataset(bundle: DatasetBundle, correlation_threshold: float = 0.85) -> DatasetProfile:
    """Computes deterministic statistical profile and health metrics for a dataset."""
    df = bundle.df
    features: List[FeatureStat] = []

    for col in bundle.feature_names:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / len(df)) if len(df) > 0 else 0.0
        unique_count = int(series.nunique())
        is_num = bool(pd.api.types.is_numeric_dtype(series))

        stat = FeatureStat(
            name=col,
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_pct=round(missing_pct, 4),
            unique_count=unique_count,
            is_numeric=is_num,
        )

        if is_num and series.notna().sum() > 0:
            stat.mean = float(series.mean())
            stat.std = float(series.std()) if len(series) > 1 else 0.0
            stat.min = float(series.min())
            stat.p25 = float(series.quantile(0.25))
            stat.p50 = float(series.median())
            stat.p75 = float(series.quantile(0.75))
            stat.max = float(series.max())

        features.append(stat)

    # Target stats
    target_series = df[bundle.target_column]
    val_counts = target_series.value_counts().to_dict()
    target_counts = {str(k): int(v) for k, v in val_counts.items()}
    total = len(target_series)
    target_proportions = {k: round(v / total, 4) for k, v in target_counts.items()}

    # Imbalance ratio: min count / max count
    counts = list(target_counts.values())
    imbalance_ratio = float(min(counts) / max(counts)) if counts and max(counts) > 0 else 1.0

    # Correlation check on numeric features
    high_corr: List[Dict[str, Any]] = []
    numeric_cols = [f.name for f in features if f.is_numeric]
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val) and val >= correlation_threshold:
                    high_corr.append({
                        "feature_1": numeric_cols[i],
                        "feature_2": numeric_cols[j],
                        "correlation": round(float(val), 4),
                    })

    return DatasetProfile(
        dataset_name=bundle.source,
        row_count=len(df),
        column_count=len(df.columns),
        target_column=bundle.target_column,
        features=features,
        target_counts=target_counts,
        target_proportions=target_proportions,
        imbalance_ratio=round(imbalance_ratio, 4),
        high_correlation_pairs=high_corr,
        fingerprint_sha256=bundle.fingerprint_sha256,
    )
