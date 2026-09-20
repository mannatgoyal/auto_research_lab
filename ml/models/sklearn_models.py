"""Scikit-learn model implementations: Logistic Regression and Random Forest."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.models.base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Deterministic wrapper around sklearn LogisticRegression."""

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None, seed: int = 42):
        self._hp = hyperparameters or {}
        self._seed = seed
        self._model = LogisticRegression(
            C=float(self._hp.get("C", 1.0)),
            max_iter=int(self._hp.get("max_iter", 1000)),
            solver=str(self._hp.get("solver", "lbfgs")),
            random_state=self._seed,
        )
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "logistic_regression"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> LogisticRegressionModel:
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 1:
            # In case only one class is present in training data
            probs = np.hstack([1.0 - probs, probs])
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_name,
            "seed": self._seed,
            "hyperparameters": self._hp,
        }


class RandomForestModel(BaseModel):
    """Deterministic wrapper around sklearn RandomForestClassifier."""

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None, seed: int = 42):
        self._hp = hyperparameters or {}
        self._seed = seed
        self._model = RandomForestClassifier(
            n_estimators=int(self._hp.get("n_estimators", 100)),
            max_depth=self._hp.get("max_depth", None),
            min_samples_split=int(self._hp.get("min_samples_split", 2)),
            min_samples_leaf=int(self._hp.get("min_samples_leaf", 1)),
            random_state=self._seed,
            n_jobs=1,  # Fixed single-thread for strict deterministic reproducibility
        )
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "random_forest"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> RandomForestModel:
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 1:
            probs = np.hstack([1.0 - probs, probs])
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_name,
            "seed": self._seed,
            "hyperparameters": self._hp,
        }
