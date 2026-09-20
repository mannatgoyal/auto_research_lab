"""Unit tests for dataset loading, fingerprinting, and profiling."""

import pytest
from ml.config import DatasetConfig
from ml.datasets.loader import compute_dataframe_fingerprint, load_dataset
from ml.datasets.profiler import profile_dataset


def test_load_breast_cancer():
    cfg = DatasetConfig(name="breast_cancer")
    bundle = load_dataset(cfg)

    assert bundle.num_rows == 569
    assert bundle.num_features == 30
    assert bundle.target_column == "target"
    assert len(bundle.fingerprint_sha256) == 64
    assert set(bundle.df[bundle.target_column].unique()) == {0, 1}


def test_fingerprint_reproducibility():
    cfg = DatasetConfig(name="breast_cancer")
    bundle1 = load_dataset(cfg)
    bundle2 = load_dataset(cfg)

    assert bundle1.fingerprint_sha256 == bundle2.fingerprint_sha256


def test_profile_dataset():
    cfg = DatasetConfig(name="breast_cancer")
    bundle = load_dataset(cfg)
    profile = profile_dataset(bundle)

    assert profile.row_count == 569
    assert profile.column_count == 31
    assert len(profile.features) == 30
    assert "0" in profile.target_counts
    assert "1" in profile.target_counts
    assert profile.imbalance_ratio > 0.0
    assert len(profile.high_correlation_pairs) > 0


def test_invalid_dataset_name_raises():
    cfg = DatasetConfig(name="non_existent_dataset_xyz")
    with pytest.raises(ValueError, match="Unknown or missing dataset"):
        load_dataset(cfg)
