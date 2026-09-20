"""Deterministic evaluation metric calculation for tabular binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ConfusionMatrixData:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @property
    def matrix(self) -> List[List[int]]:
        return [
            [self.true_negative, self.false_positive],
            [self.false_negative, self.true_positive],
        ]


@dataclass
class EvaluationMetrics:
    """Structured container for deterministic machine learning metrics."""
    accuracy: float
    balanced_accuracy: float
    precision_binary: float
    recall_binary: float
    f1_binary: float
    f1_macro: float
    f1_weighted: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    log_loss_val: Optional[float]
    confusion_matrix: ConfusionMatrixData

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["confusion_matrix"] = self.confusion_matrix.matrix
        return data


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> EvaluationMetrics:
    """Computes a deterministic suite of classification metrics.

    Args:
        y_true: 1D array of ground truth labels (0 or 1).
        y_pred: 1D array of binary model predictions (0 or 1).
        y_proba: Optional 2D array of class probabilities shape (N, 2).

    Returns:
        EvaluationMetrics object with rounded deterministic numerical values.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_bin = float(f1_score(y_true, y_pred, zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    cm_data = ConfusionMatrixData(
        true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp
    )

    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None
    log_loss_metric: Optional[float] = None

    if y_proba is not None:
        try:
            # Positive class probability (index 1)
            pos_proba = y_proba[:, 1]
            if len(np.unique(y_true)) > 1:
                roc_auc_val = float(roc_auc_score(y_true, pos_proba))
                pr_auc_val = float(average_precision_score(y_true, pos_proba))
                log_loss_metric = float(log_loss(y_true, y_proba))
        except Exception:
            pass

    return EvaluationMetrics(
        accuracy=round(acc, 5),
        balanced_accuracy=round(bal_acc, 5),
        precision_binary=round(prec, 5),
        recall_binary=round(rec, 5),
        f1_binary=round(f1_bin, 5),
        f1_macro=round(f1_mac, 5),
        f1_weighted=round(f1_wt, 5),
        roc_auc=round(roc_auc_val, 5) if roc_auc_val is not None else None,
        pr_auc=round(pr_auc_val, 5) if pr_auc_val is not None else None,
        log_loss_val=round(log_loss_metric, 5) if log_loss_metric is not None else None,
        confusion_matrix=cm_data,
    )
