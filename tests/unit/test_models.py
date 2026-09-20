"""Unit tests for model wrappers and training interface."""

import numpy as np
import pytest
from ml.config import ModelConfig
from ml.models.base import create_model


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest", "xgboost", "torch_mlp"])
def test_model_lifecycle(model_name, sample_data):
    X, y = sample_data
    cfg = ModelConfig(name=model_name)
    model = create_model(cfg, seed=42)

    assert model.model_name == model_name

    # Train
    model.fit(X[:80], y[:80], X_val=X[80:], y_val=y[80:])

    # Predict
    preds = model.predict(X[80:])
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Predict proba
    probs = model.predict_proba(X[80:])
    assert probs.shape == (20, 2)
    # Probabilities must sum to 1.0
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_seed_reproducibility(sample_data):
    X, y = sample_data
    cfg = ModelConfig(name="random_forest", hyperparameters={"n_estimators": 20})

    m1 = create_model(cfg, seed=777).fit(X, y)
    m2 = create_model(cfg, seed=777).fit(X, y)

    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)

    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_torch_mlp_single_sample_inference(sample_data):
    """Verifies that PyTorch MLP cleanly handles a single 1D feature vector."""
    X, y = sample_data
    model = create_model(ModelConfig(name="torch_mlp")).fit(X, y)

    single_x = X[0]  # Shape (D,)
    pred = model.predict(single_x)
    proba = model.predict_proba(single_x)

    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    np.testing.assert_allclose(np.sum(proba), 1.0, atol=1e-5)


def test_torch_mlp_batch_size_one_safety():
    """Verifies that dataset length where len % batch_size == 1 does not crash BatchNorm1d."""
    np.random.seed(42)
    # 33 samples with batch_size=32 leaves 1 sample in the final batch
    X = np.random.randn(33, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=33)

    model = create_model(
        ModelConfig(name="torch_mlp", hyperparameters={"batch_size": 32, "max_epochs": 2})
    )
    # Must train without raising ValueError from BatchNorm1d
    model.fit(X, y)
    assert model.predict(X[:2]).shape == (2,)
