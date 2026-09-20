"""Leak-free preprocessing pipelines utilizing scikit-learn ColumnTransformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from ml.config import PreprocessingConfig
from ml.preprocessing.splitters import SplitData


@dataclass
class PreprocessedData:
    """Preprocessed numpy arrays ready for model consumption."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    transformed_feature_names: List[str]


@dataclass
class PreprocessorPipeline:
    """Wrapper holding fitted transformers and preprocessing metadata."""
    transformer: ColumnTransformer
    config: PreprocessingConfig
    numeric_features: List[str]
    categorical_features: List[str]
    transformed_feature_names: List[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms unseen data using the pipeline fitted strictly on training data."""
        return self.transformer.transform(df)


def build_and_fit_pipeline(
    split_data: SplitData, config: PreprocessingConfig
) -> Tuple[PreprocessorPipeline, PreprocessedData]:
    """Constructs and fits a preprocessing pipeline strictly on the training partition.

    Enforces:
    - Zero data leakage: Parameters (mean, std, median, categories) estimated ONLY from train_df.
    - Deterministic transformation of validation and test partitions.
    """
    train_df = split_data.train_df
    val_df = split_data.val_df
    test_df = split_data.test_df
    target_col = split_data.target_column
    feature_names = split_data.feature_names

    # Detect column types from training data only
    X_train_df = train_df[feature_names]
    numeric_features = [
        col for col in feature_names if pd.api.types.is_numeric_dtype(X_train_df[col])
    ]
    categorical_features = [
        col for col in feature_names if col not in numeric_features
    ]

    transformers = []

    # 1. Numeric pipeline
    if numeric_features:
        numeric_steps = []
        imputer_strategy = "constant" if config.numeric_imputer == "zero" else config.numeric_imputer
        fill_val = 0.0 if config.numeric_imputer == "zero" else None
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy=imputer_strategy, fill_value=fill_val))
        )

        if config.scaler == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif config.scaler == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
        elif config.scaler == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        # "none" leaves data unscaled

        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))

    # 2. Categorical pipeline
    if categorical_features and config.encode_categoricals:
        cat_steps = [
            (
                "imputer",
                SimpleImputer(strategy=config.categorical_imputer, fill_value="missing"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
        transformers.append(("categorical", Pipeline(cat_steps), categorical_features))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    # CRITICAL: Fit ONLY on training features
    column_transformer.fit(X_train_df)

    # Retrieve transformed feature names
    try:
        out_names = list(column_transformer.get_feature_names_out())
    except Exception:
        out_names = [f"feat_{i}" for i in range(column_transformer.transform(X_train_df[:1]).shape[1])]

    # Transform all partitions
    X_train = column_transformer.transform(X_train_df)
    y_train = train_df[target_col].to_numpy()

    if len(val_df) > 0:
        X_val = column_transformer.transform(val_df[feature_names])
        y_val = val_df[target_col].to_numpy()
    else:
        X_val = np.empty((0, X_train.shape[1]))
        y_val = np.empty((0,))

    X_test = column_transformer.transform(test_df[feature_names])
    y_test = test_df[target_col].to_numpy()

    # Cast to float32 for model training stability
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    preprocessed = PreprocessedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        transformed_feature_names=out_names,
    )

    pipeline = PreprocessorPipeline(
        transformer=column_transformer,
        config=config,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        transformed_feature_names=out_names,
    )

    return pipeline, preprocessed
