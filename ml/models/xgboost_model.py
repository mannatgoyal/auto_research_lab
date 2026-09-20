"""Deterministic wrapper around XGBoost Classifier."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import xgboost as xgb

from ml.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Deterministic wrapper around XGBoost Classifier."""

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None, seed: int = 42):
        self._hp = hyperparameters or {}
        self._seed = seed
        self._model = xgb.XGBClassifier(
            n_estimators=int(self._hp.get("n_estimators", 100)),
            max_depth=int(self._hp.get("max_depth", 4)),
            learning_rate=float(self._hp.get("learning_rate", 0.1)),
            subsample=float(self._hp.get("subsample", 1.0)),
            colsample_bytree=float(self._hp.get("colsample_bytree", 1.0)),
            reg_alpha=float(self._hp.get("reg_alpha", 0.0)),
            reg_lambda=float(self._hp.get("reg_lambda", 1.0)),
            eval_metric=str(self._hp.get("eval_metric", "logloss")),
            random_state=self._seed,
            n_jobs=1,  # Single-threaded to enforce reproducibility across machines
            tree_method="hist",
        )
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "xgboost"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> XGBoostModel:
        fit_params = {}
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self._model.fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self._model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_name,
            "seed": self._seed,
            "hyperparameters": self._hp,
        }
