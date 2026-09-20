"""Model loading abstraction and inference pipeline construction from MLflow artifacts."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from api.config import ServingConfig
from api.schemas import CLASS_LABELS, FEATURE_NAMES_30
from ml.preprocessing.pipeline import PreprocessorPipeline

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Encapsulates a loaded model estimator paired with its training-time PreprocessorPipeline."""

    def __init__(
        self,
        preprocessor: PreprocessorPipeline,
        model: Any,
        metadata: Dict[str, Any],
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.metadata = metadata

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", "unknown"))

    @property
    def run_id(self) -> str:
        return str(self.metadata.get("run_id", "unknown"))

    @property
    def parent_run_id(self) -> Optional[str]:
        return self.metadata.get("parent_run_id")

    def predict(self, df: pd.DataFrame) -> Tuple[int, Optional[Dict[str, float]]]:
        """Transforms raw input features and computes class prediction and probabilities.

        Guarantee:
        Uses the exact fitted PreprocessorPipeline from training.
        """
        # 1. Transform using the exact fitted ColumnTransformer
        X_preprocessed = self.preprocessor.transform(df)
        X_array = np.asarray(X_preprocessed, dtype=np.float32)

        # 2. Run model prediction
        if hasattr(self.model, "predict"):
            pred = self.model.predict(X_array)
        else:
            raise RuntimeError(f"Loaded model {type(self.model)} has no predict() method.")

        predicted_class = int(np.asarray(pred).flatten()[0])

        # 3. Compute probabilities if supported
        probabilities: Optional[Dict[str, float]] = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_array)
            probs_arr = np.asarray(probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                prob_neg = float(probs_arr[0, 0])
                prob_pos = float(probs_arr[0, 1])
                probabilities = {
                    CLASS_LABELS.get(0, "class_0"): round(prob_neg, 6),
                    CLASS_LABELS.get(1, "class_1"): round(prob_pos, 6),
                }

        return predicted_class, probabilities


class ModelLoader:
    """Loads model artifacts and parent preprocessing pipelines from an MLflow tracking store."""

    def __init__(self, tracking_uri: Optional[str] = None) -> None:
        self.tracking_uri = tracking_uri or "sqlite:///mlflow.db"
        # Ensure filesystem store compatibility in MLflow 3.x
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

    def load_pipeline(self, config: ServingConfig) -> InferencePipeline:
        """Resolves child model run and parent preprocessing pipeline to construct an InferencePipeline."""
        run_id = config.run_id
        logger.info("Loading child model run: %s from %s", run_id, self.tracking_uri)

        try:
            child_run = self.client.get_run(run_id)
        except Exception as e:
            raise ValueError(f"Could not retrieve MLflow run with ID '{run_id}': {e}") from e

        # 1. Resolve Parent Run for the Preprocessing Pipeline
        parent_run_id = child_run.data.tags.get("parent_run_id")
        if not parent_run_id:
            raise ValueError(
                f"Child run '{run_id}' is missing required 'parent_run_id' tag. "
                "Cannot resolve experiment-level preprocessing pipeline."
            )

        try:
            parent_run = self.client.get_run(parent_run_id)
        except Exception as e:
            raise ValueError(
                f"Could not retrieve MLflow parent run '{parent_run_id}': {e}"
            ) from e

        # 2. Download and deserialize the fitted PreprocessorPipeline
        try:
            prep_artifact_path = mlflow.artifacts.download_artifacts(
                run_id=parent_run_id,
                artifact_path="preprocessing/pipeline.joblib",
                tracking_uri=self.tracking_uri,
            )
            preprocessor: PreprocessorPipeline = joblib.load(prep_artifact_path)
            if not isinstance(preprocessor, PreprocessorPipeline):
                raise TypeError(
                    f"Expected PreprocessorPipeline instance, got {type(preprocessor)}"
                )
            logger.info("Successfully loaded fitted PreprocessorPipeline from parent run %s", parent_run_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load fitted preprocessing pipeline from parent run '{parent_run_id}': {e}"
            ) from e

        # 3. Load the candidate model artifact
        model_name = child_run.data.tags.get(
            "model_name", child_run.data.params.get("model_type", "model")
        )
        model = self._load_model_artifact(child_run)

        # 4. Extract Lineage Metadata
        exp = self.client.get_experiment(child_run.info.experiment_id)
        metadata = {
            "model_name": model_name,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "experiment_name": exp.name if exp else "unknown",
            "model_type": child_run.data.params.get("model_type", model_name),
            "dataset_fingerprint": child_run.data.tags.get(
                "dataset_fingerprint", parent_run.data.tags.get("dataset_fingerprint", "unknown")
            ),
            "config_hash_sha256": child_run.data.tags.get(
                "config_hash_sha256", parent_run.data.tags.get("config_hash_sha256", "unknown")
            ),
            "git_commit": child_run.data.tags.get(
                "git_commit", parent_run.data.tags.get("git_commit", "untracked")
            ),
            "expected_features": preprocessor.numeric_features or FEATURE_NAMES_30,
            "transformed_features_count": len(preprocessor.transformed_feature_names),
        }

        return InferencePipeline(
            preprocessor=preprocessor,
            model=model,
            metadata=metadata,
        )

    def _load_model_artifact(self, child_run: Any) -> Any:
        """Loads the model estimator from the child run's direct file artifact or flavor."""
        run_id = child_run.info.run_id
        model_name = child_run.data.tags.get(
            "model_name", child_run.data.params.get("model_type", "model")
        )

        # Strategy A: Load portable file artifact under model/
        try:
            if model_name in ("logistic_regression", "random_forest"):
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="model/model.joblib",
                    tracking_uri=self.tracking_uri,
                )
                return joblib.load(local_path)
            elif model_name == "xgboost":
                import xgboost as xgb
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="model/model.json",
                    tracking_uri=self.tracking_uri,
                )
                booster = xgb.XGBClassifier()
                booster.load_model(local_path)
                return booster
            elif model_name == "torch_mlp":
                import torch
                from ml.models.torch_mlp import TabularMLPNetwork
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="model/model_weights.pt",
                    tracking_uri=self.tracking_uri,
                )
                state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
                # Infer hidden dimensions and input dim from state_dict
                input_dim = state_dict["net.0.weight"].shape[1]
                net = TabularMLPNetwork(input_dim=input_dim, hidden_dims=[64, 32], dropout_rate=0.0)
                net.load_state_dict(state_dict)
                net.eval()

                # Wrap PyTorch module in a lightweight predict/predict_proba adapter
                class TorchPredictorAdapter:
                    def __init__(self, torch_net: torch.nn.Module):
                        self._net = torch_net

                    def predict(self, X: np.ndarray) -> np.ndarray:
                        with torch.no_grad():
                            t = torch.tensor(X, dtype=torch.float32)
                            logits = self._net(t)
                            preds = (logits.squeeze(-1) > 0.0).long().numpy()
                            return preds

                    def predict_proba(self, X: np.ndarray) -> np.ndarray:
                        with torch.no_grad():
                            t = torch.tensor(X, dtype=torch.float32)
                            logits = self._net(t)
                            p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
                            p0 = 1.0 - p1
                            return np.column_stack([p0, p1])

                return TorchPredictorAdapter(net)
        except Exception as e:
            logger.warning("Direct file artifact load failed for %s (%s). Attempting flavor loader.", run_id, e)

        # Strategy B: Fall back to MLflow native flavor
        try:
            if model_name in ("logistic_regression", "random_forest"):
                return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            elif model_name == "xgboost":
                return mlflow.xgboost.load_model(f"runs:/{run_id}/model")
            elif model_name == "torch_mlp":
                return mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact for run '{run_id}': {e}") from e

        raise ValueError(f"Unsupported model architecture '{model_name}'")
