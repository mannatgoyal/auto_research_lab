"""Unit tests for deterministic metric evaluation."""

import numpy as np
from ml.evaluation.metrics import evaluate_predictions


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.05, 0.95],
    ])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.precision_binary == 1.0
    assert metrics.recall_binary == 1.0
    assert metrics.f1_binary == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix.true_negative == 2
    assert metrics.confusion_matrix.true_positive == 3
    assert metrics.confusion_matrix.false_positive == 0
    assert metrics.confusion_matrix.false_negative == 0


def test_imperfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])  # TN=1, FP=1, FN=1, TP=1

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.accuracy == 0.5
    assert metrics.balanced_accuracy == 0.5
    assert metrics.precision_binary == 0.5
    assert metrics.recall_binary == 0.5
    assert metrics.f1_binary == 0.5
    assert metrics.confusion_matrix.matrix == [[1, 1], [1, 1]]


def test_metric_serialization():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    metrics = evaluate_predictions(y_true, y_pred)
    d = metrics.to_dict()

    assert isinstance(d, dict)
    assert "accuracy" in d
    assert "f1_macro" in d
    assert "confusion_matrix" in d
    assert d["confusion_matrix"] == [[1, 0], [0, 1]]
