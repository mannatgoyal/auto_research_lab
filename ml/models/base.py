"""Base interface for all machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from ml.config import ModelConfig


class BaseModel(ABC):
    """Abstract base class for all deterministic model wrappers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns canonical name of the model."""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> BaseModel:
        """Trains the model on training data, optionally monitoring validation performance."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates binary class predictions (0 or 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generates class probability distributions of shape (N, 2)."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Returns hyperparameters and configuration."""
        pass


def create_model(model_config: ModelConfig, seed: int = 42) -> BaseModel:
    """Factory creating an un-fitted model instance from ModelConfig."""
    name = model_config.name.lower()

    if name == "logistic_regression":
        from ml.models.sklearn_models import LogisticRegressionModel
        return LogisticRegressionModel(hyperparameters=model_config.hyperparameters, seed=seed)

    elif name == "random_forest":
        from ml.models.sklearn_models import RandomForestModel
        return RandomForestModel(hyperparameters=model_config.hyperparameters, seed=seed)

    elif name == "xgboost":
        from ml.models.xgboost_model import XGBoostModel
        return XGBoostModel(hyperparameters=model_config.hyperparameters, seed=seed)

    elif name == "torch_mlp":
        from ml.models.torch_mlp import TorchMLPModel
        return TorchMLPModel(hyperparameters=model_config.hyperparameters, seed=seed)

    else:
        raise ValueError(f"Unsupported model architecture '{name}'")
