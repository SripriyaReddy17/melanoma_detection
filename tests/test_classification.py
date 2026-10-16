# Tests for classification.

from app.services.classification import (
    MODEL_BACKBONE,
    _binary_melanoma_probability,
    _heuristic_risk,
)


def test_model_uses_vit_backbone():
    assert MODEL_BACKBONE == "google/vit-base-patch16-224"


def test_binary_melanoma_probability_uses_melanoma_class_index():
    probs = {"non_mel": 0.72, "mel": 0.28}
    assert _binary_melanoma_probability(probs) == 0.28


def test_suspicious_border_and_large_diameter_increase_melanoma_risk():
    features = {
        "asymmetry": 0.16,
        "border_irregularity": 0.58,
        "color_index": 0.29,
        "diameter_mm": 31.5,
        "evolution_change_detected": False,
    }

    risk = _heuristic_risk(features)

    assert risk >= 0.5
