"""Deterministic PyTorch Multi-Layer Perceptron (MLP) for tabular classification."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.base import BaseModel


class TabularMLPNetwork(nn.Module):
    """Feedforward neural network with BatchNorm and Dropout for tabular features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPModel(BaseModel):
    """Deterministic wrapper around a PyTorch Tabular MLP."""

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None, seed: int = 42):
        self._hp = hyperparameters or {}
        self._seed = seed
        self._hidden_dims = list(self._hp.get("hidden_dims", [64, 32]))
        self._lr = float(self._hp.get("lr", 1e-3))
        self._weight_decay = float(self._hp.get("weight_decay", 1e-4))
        self._batch_size = int(self._hp.get("batch_size", 32))
        self._max_epochs = int(self._hp.get("max_epochs", 100))
        self._dropout_rate = float(self._hp.get("dropout_rate", 0.1))
        self._patience = int(self._hp.get("early_stopping_patience", 10))
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and self._hp.get("device") == "cuda" else "cpu"
        )
        self._net: Optional[TabularMLPNetwork] = None
        self._input_dim: Optional[int] = None
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        return "torch_mlp"

    def _set_deterministic_seeds(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> TorchMLPModel:
        self._set_deterministic_seeds()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        self._input_dim = X_arr.shape[1]

        self._net = TabularMLPNetwork(
            input_dim=self._input_dim,
            hidden_dims=self._hidden_dims,
            num_classes=2,
            dropout_rate=self._dropout_rate,
        ).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )

        train_dataset = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        # Drop last batch if it contains only 1 sample to avoid BatchNorm1d crash
        drop_last = len(train_dataset) > self._batch_size and (len(train_dataset) % self._batch_size == 1)
        g = torch.Generator()
        g.manual_seed(self._seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=g,
        )

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            val_X_tensor = torch.tensor(X_val, dtype=torch.float32).to(self._device)
            val_y_tensor = torch.tensor(y_val, dtype=torch.long).to(self._device)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self._max_epochs):
            self._net.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                outputs = self._net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validation loss check
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(val_X_tensor)
                    current_loss = float(criterion(val_logits, val_y_tensor).item())

                if current_loss < best_loss - 1e-4:
                    best_loss = current_loss
                    best_state = copy.deepcopy(self._net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        break

        # Restore best weights if early stopping was active
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._net.eval()
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        self._net.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        with torch.no_grad():
            x_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self._device)
            logits = self._net(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_name,
            "seed": self._seed,
            "device": str(self._device),
            "hyperparameters": self._hp,
        }
