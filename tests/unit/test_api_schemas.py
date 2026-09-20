"""Unit tests for FastAPI request and response Pydantic schemas."""

import math
import pytest
from pydantic import ValidationError

from api.schemas import BreastCancerFeatures, FEATURE_NAMES_30, PredictionRequest


@pytest.fixture
def valid_sample_features():
    """Generates a valid dictionary with all 30 features."""
    return {feat: float(i + 1) for i, feat in enumerate(FEATURE_NAMES_30)}


def test_valid_30_features(valid_sample_features):
    features = BreastCancerFeatures(**valid_sample_features)
    assert features.mean_radius == 1.0
    assert features.worst_fractal_dimension == 30.0

    df = features.to_dataframe()
    assert df.shape == (1, 30)
    assert list(df.columns) == FEATURE_NAMES_30
    assert df["mean radius"].iloc[0] == 1.0


def test_missing_feature_rejected(valid_sample_features):
    # Remove one required feature
    del valid_sample_features["worst fractal dimension"]
    with pytest.raises(ValidationError) as exc_info:
        BreastCancerFeatures(**valid_sample_features)
    assert "Field required" in str(exc_info.value)


def test_extra_unexpected_field_rejected(valid_sample_features):
    # Add an unexpected field
    valid_sample_features["unknown_extra_metric"] = 999.0
    with pytest.raises(ValidationError) as exc_info:
        BreastCancerFeatures(**valid_sample_features)
    assert "Extra inputs are not permitted" in str(exc_info.value)


def test_invalid_type_rejected(valid_sample_features):
    valid_sample_features["mean radius"] = "not_a_number"
    with pytest.raises(ValidationError) as exc_info:
        BreastCancerFeatures(**valid_sample_features)
    assert "Input should be a valid number" in str(exc_info.value)


def test_nan_value_rejected(valid_sample_features):
    valid_sample_features["mean area"] = float("nan")
    with pytest.raises(ValidationError) as exc_info:
        BreastCancerFeatures(**valid_sample_features)
    assert "NaN and Inf are forbidden" in str(exc_info.value)


def test_inf_value_rejected(valid_sample_features):
    valid_sample_features["mean area"] = float("inf")
    with pytest.raises(ValidationError) as exc_info:
        BreastCancerFeatures(**valid_sample_features)
    assert "NaN and Inf are forbidden" in str(exc_info.value)


def test_prediction_request_supports_flat_and_nested(valid_sample_features):
    # Test flat format
    req1 = PredictionRequest.model_validate(valid_sample_features)
    assert req1.features.mean_radius == 1.0

    # Test nested format
    req2 = PredictionRequest.model_validate({"features": valid_sample_features})
    assert req2.features.mean_radius == 1.0
