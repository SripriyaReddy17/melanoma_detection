"""Compute basic classification metrics for model evaluation."""

from __future__ import annotations

import argparse
from collections import Counter


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, F1 and support for a multiclass problem."""
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))

    confusion = {label: {pred: 0 for pred in labels} for label in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] = confusion[t].get(p, 0) + 1

    metrics = {
        "accuracy": _safe_divide(correct, total),
        "labels": labels,
        "support": {label: sum(1 for t in y_true if t == label) for label in labels},
        "confusion_matrix": confusion,
    }

    per_class = {}
    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in labels if other != label)
        fn = sum(confusion[label].get(other, 0) for other in labels if other != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": metrics["support"][label],
        }

    metrics["per_class"] = per_class
    return metrics


def _parse_labels(values):
    return [int(v) if str(v).lstrip("-").isdigit() else str(v) for v in values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute classification metrics.")
    parser.add_argument("--true", nargs="+", required=True, help="Ground-truth labels.")
    parser.add_argument("--pred", nargs="+", required=True, help="Predicted labels.")
    args = parser.parse_args()

    y_true = _parse_labels(args.true)
    y_pred = _parse_labels(args.pred)

    if len(y_true) != len(y_pred):
        raise ValueError("Number of true labels and predicted labels must match.")

    metrics = compute_metrics(y_true, y_pred)
    print(metrics)
