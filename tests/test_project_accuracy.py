from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import compute_metrics


def test_project_accuracy_metric_is_correct_for_known_labels():
    y_true = [0, 1, 1, 0, 2, 2, 1]
    y_pred = [0, 1, 0, 0, 2, 1, 1]

    result = compute_metrics(y_true, y_pred)

    assert result["accuracy"] == 5 / 7
    assert result["support"] == {0: 2, 1: 3, 2: 2}
    assert result["per_class"][1]["f1"] == 2 / 3


def test_project_accuracy_metric_returns_zero_for_all_wrong_predictions():
    y_true = [0, 1, 2]
    y_pred = [2, 0, 1]

    result = compute_metrics(y_true, y_pred)

    assert result["accuracy"] == 0.0
    assert result["per_class"][0]["precision"] == 0.0
    assert result["per_class"][1]["recall"] == 0.0
