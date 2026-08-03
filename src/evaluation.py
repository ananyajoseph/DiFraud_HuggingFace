"""Leakage-safe threshold selection and imbalanced classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)


def choose_threshold(y_validation, p_validation, objective: str = "f1") -> float:
    y, p = np.asarray(y_validation), np.asarray(p_validation)
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return 0.5
    if objective == "f1":
        score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    elif objective == "balanced_accuracy":
        score = np.array([balanced_accuracy_score(y, p >= t) for t in thresholds])
    else:
        raise ValueError("objective must be 'f1' or 'balanced_accuracy'.")
    return float(thresholds[int(np.nanargmax(score))])


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def recall_at_precision(y, p, target: float = 0.9) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    eligible = recall[precision >= target]
    return float(eligible.max()) if len(eligible) else float("nan")


def classification_metrics(y, p, threshold: float, alert_percentile: float = 0.95) -> dict[str, float]:
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    budget_t = np.quantile(p, alert_percentile)
    budget_pred = p >= budget_t
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": average_precision_score(y, p),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "recall_at_90_precision": recall_at_precision(y, p),
        "precision_at_5pct_alert_budget": precision_score(y, budget_pred, zero_division=0),
        "false_positives_per_1000_negatives": 1000 * fp / max(tn + fp, 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

