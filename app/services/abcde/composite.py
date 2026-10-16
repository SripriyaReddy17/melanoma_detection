"""Composite ABCDE melanoma risk scoring."""

from __future__ import annotations


def compute_abcde_score(features: dict) -> dict:
    """Aggregate normalized ABCDE features into a single melanoma-risk score.

    This is intentionally interpretable and lightweight so it can be used in
    reporting and risk ranking without pretending to be a clinically validated
    diagnostic model.
    """
    if not isinstance(features, dict):
        return {
            "score": 0.0,
            "label": "low",
            "reason": "No ABCDE feature dictionary was provided.",
        }

    asym = float(features.get("asymmetry", 0.0) or 0.0)
    border = float(features.get("border_irregularity", 0.0) or 0.0)
    color = float(features.get("color_index", 0.0) or 0.0)
    diameter = features.get("diameter_mm")
    evolution = bool(features.get("evolution_change_detected", False))

    diameter_score = 0.0
    if diameter is not None:
        d = float(diameter)
        if d >= 6.0:
            diameter_score = 1.0
        elif d >= 4.0:
            diameter_score = 0.6
        elif d >= 2.0:
            diameter_score = 0.25

    # Weight the clinically relevant components more strongly.
    score = (
        0.24 * asym
        + 0.22 * border
        + 0.18 * color
        + 0.18 * diameter_score
        + 0.18 * (1.0 if evolution else 0.0)
    )
    score = max(0.0, min(1.0, float(score)))

    if score >= 0.70:
        label = "high"
    elif score >= 0.45:
        label = "moderate"
    else:
        label = "low"

    return {
        "score": round(score, 4),
        "label": label,
        "components": {
            "asymmetry": round(asym, 4),
            "border_irregularity": round(border, 4),
            "color_index": round(color, 4),
            "diameter_mm": diameter,
            "evolution_change_detected": evolution,
        },
        "reason": (
            "Composite ABCDE risk is a heuristic summary of lesion asymmetry, "
            "border irregularity, color heterogeneity, lesion size, and longitudinal change. "
            "It is interpretive and should not replace clinical assessment."
        ),
    }
