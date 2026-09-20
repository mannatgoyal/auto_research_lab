"""Unit tests for deterministic data splitting."""

import pytest
from ml.config import DatasetConfig, SplitConfig
from ml.datasets.loader import load_dataset
from ml.preprocessing.splitters import create_splits


@pytest.fixture
def dataset_bundle():
    return load_dataset(DatasetConfig(name="breast_cancer"))


def test_create_splits_no_overlap(dataset_bundle):
    cfg = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.1, seed=42)
    split = create_splits(dataset_bundle, cfg)

    train_idx = set(split.train_df.index)
    val_idx = set(split.val_df.index)
    test_idx = set(split.test_df.index)

    # 1. No overlap
    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0

    # 2. Total row count matches
    assert split.total_rows == dataset_bundle.num_rows

    # 3. Stratification preserved
    train_ratio = split.train_df[split.target_column].mean()
    test_ratio = split.test_df[split.target_column].mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_split_determinism(dataset_bundle):
    cfg1 = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.1, seed=99)
    cfg2 = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.1, seed=99)

    split1 = create_splits(dataset_bundle, cfg1)
    split2 = create_splits(dataset_bundle, cfg2)

    assert split1.train_sha256 == split2.train_sha256
    assert split1.val_sha256 == split2.val_sha256
    assert split1.test_sha256 == split2.test_sha256
    assert list(split1.train_df.index) == list(split2.train_df.index)


def test_different_seeds_produce_different_splits(dataset_bundle):
    cfg1 = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.1, seed=1)
    cfg2 = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.1, seed=2)

    split1 = create_splits(dataset_bundle, cfg1)
    split2 = create_splits(dataset_bundle, cfg2)

    assert split1.train_sha256 != split2.train_sha256


def test_split_with_zero_validation(dataset_bundle):
    """Verifies splitting when val_size is 0.0."""
    cfg = SplitConfig(strategy="stratified", test_size=0.2, val_size=0.0, seed=42)
    split = create_splits(dataset_bundle, cfg)

    assert split.val_rows == 0
    assert len(split.val_df) == 0
    assert split.train_rows + split.test_rows == dataset_bundle.num_rows
    assert split.val_sha256 == "empty"
