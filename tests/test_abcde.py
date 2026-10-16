# Tests for ABCDE feature extraction.

import cv2
import numpy as np

from app.services.abcde.asymmetry import compute_asymmetry
from app.services.abcde.color import compute_color_variation
from app.services.abcde.composite import compute_abcde_score
from app.services.abcde.diameter import compute_diameter_mm
from app.services.abcde.evolution import compute_evolution


def test_diameter_is_marked_unavailable_without_calibration():
    result = compute_diameter_mm(mask=None, pixels_per_mm=None)

    assert result["available"] is False
    assert result["max_diameter_mm"] is None
    assert "calibration" in result["reason"].lower()


def test_evolution_requires_reliable_longitudinal_data():
    previous = {"diameter_mm": None, "asymmetry": 0.4}
    current = {"diameter_mm": None, "asymmetry": 0.5}

    result = compute_evolution(previous, current)

    assert result["available"] is False
    assert result["change_detected"] is False
    assert "longitudinal" in result["reason"].lower() or "follow-up" in result["reason"].lower()


def test_melanoma_like_lesion_has_notable_asymmetry_and_color_variation():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    mask = np.zeros((220, 220), dtype=np.uint8)

    cv2.ellipse(mask, (110, 110), (80, 60), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (140, 110), (30, 20), 0, 0, 360, 0, -1)

    for y in range(220):
        for x in range(220):
            if mask[y, x] > 0:
                image[y, x] = [30, 80, 100]

    image[50:120, 90:140] = [30, 120, 160]
    image[120:180, 110:160] = [15, 40, 70]

    asymmetry = compute_asymmetry(image, mask)
    color = compute_color_variation(image, mask)

    assert asymmetry > 0.20
    assert color["index"] > 0.30


def test_abcde_composite_score_rises_for_melanoma_like_features():
    features = {
        "asymmetry": 0.82,
        "border_irregularity": 0.76,
        "color": {"index": 0.74},
        "diameter_mm": 9.5,
        "evolution_change_detected": True,
    }

    score = compute_abcde_score(features)

    assert score["score"] > 0.60
    assert score["label"] == "high"
