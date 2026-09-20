"""Unit tests for leak-free preprocessing pipeline."""

import numpy as np
import pytest
from ml.config import DatasetConfig, PreprocessingConfig, SplitConfig
from ml.datasets.loader import load_dataset
from ml.preprocessing.pipeline import build_and_fit_pipeline
from ml.preprocessing.splitters import SplitData, create_splits


@pytest.fixture
def split_data():
    bundle = load_dataset(DatasetConfig(name="breast_cancer"))
    return create_splits(bundle, SplitConfig(seed=42))


def test_preprocessing_shapes(split_data):
    cfg = PreprocessingConfig(scaler="standard", numeric_imputer="median")
    pipeline, preprocessed = build_and_fit_pipeline(split_data, cfg)

    # Output dimensions check
    assert preprocessed.X_train.shape[0] == split_data.train_rows
    assert preprocessed.X_val.shape[0] == split_data.val_rows
    assert preprocessed.X_test.shape[0] == split_data.test_rows
    assert preprocessed.X_train.shape[1] == len(split_data.feature_names)
    assert len(preprocessed.transformed_feature_names) == preprocessed.X_train.shape[1]


def test_standard_scaler_train_normalization(split_data):
    cfg = PreprocessingConfig(scaler="standard")
    _, preprocessed = build_and_fit_pipeline(split_data, cfg)

    # Train mean should be ~0, std should be ~1
    train_means = np.mean(preprocessed.X_train, axis=0)
    train_stds = np.std(preprocessed.X_train, axis=0)

    np.testing.assert_allclose(train_means, 0.0, atol=1e-5)
    np.testing.assert_allclose(train_stds, 1.0, atol=1e-4)


def test_pipeline_transform_unseen_data(split_data):
    cfg = PreprocessingConfig(scaler="minmax")
    pipeline, _ = build_and_fit_pipeline(split_data, cfg)

    # Transform test dataframe directly
    test_transformed = pipeline.transform(split_data.test_df[split_data.feature_names])
    assert test_transformed.shape[0] == split_data.test_rows


def test_zero_data_leakage_mathematical_proof(split_data):
    """Proves mathematically that modifications to test_df cannot alter training pipeline parameters."""
    cfg = PreprocessingConfig(scaler="standard")
    
    # 1. Fit pipeline on clean split
    pipeline1, _ = build_and_fit_pipeline(split_data, cfg)
    scaler1 = pipeline1.transformer.named_transformers_["numeric"].named_steps["scaler"]
    original_means = scaler1.mean_.copy()

    # 2. Corrupt test partition with extreme outliers (1 billion)
    corrupted_test = split_data.test_df.copy()
    first_feat = split_data.feature_names[0]
    corrupted_test[first_feat] = 1_000_000_000.0

    corrupted_split = SplitData(
        train_df=split_data.train_df,
        val_df=split_data.val_df,
        test_df=corrupted_test,
        target_column=split_data.target_column,
        feature_names=split_data.feature_names,
        split_config=split_data.split_config,
        train_sha256=split_data.train_sha256,
        val_sha256=split_data.val_sha256,
        test_sha256="corrupted",
    )

    # 3. Fit pipeline on corrupted test split
    pipeline2, _ = build_and_fit_pipeline(corrupted_split, cfg)
    scaler2 = pipeline2.transformer.named_transformers_["numeric"].named_steps["scaler"]
    corrupted_test_means = scaler2.mean_.copy()

    # Means must remain identical to 12 decimal places
    np.testing.assert_allclose(original_means, corrupted_test_means, atol=1e-12)
