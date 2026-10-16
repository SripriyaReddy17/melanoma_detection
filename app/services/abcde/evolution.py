"""
Evolution (E) feature comparison for CEFM.
Complies with Rule 16 and Rule 41:
Temporal lesion comparison across follow-up visits.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def compute_evolution(
    previous_features: Optional[dict],
    current_features: dict,
) -> dict:
    """
    Compare previous and current lesion measurements.

    Evolution is available only when reliable longitudinal measurements
    exist for both the previous and current lesion.

    A dictionary containing only unavailable values or asymmetry alone
    is not sufficient to establish reliable longitudinal change.
    """

    def _unavailable(reason: str) -> dict:
        return {
            "available": False,
            "change_detected": False,
            "changes": {},
            "diameter_change": 0.0,
            "area_change": 0.0,
            "asymmetry_change": 0.0,
            "border_change": 0.0,
            "color_change": 0.0,
            "overall_change": 0.0,
            "label": "Evolution cannot be reliably assessed.",
            "description": (
                "Evolution requires reliable longitudinal measurements "
                "from previous and current evaluations of the same lesion."
            ),
            "reason": reason,
        }

    # No previous evaluation means E cannot be assessed.
    if not previous_features:
        return _unavailable("Previous image not provided.")

    if not current_features:
        return _unavailable("Current feature measurements not provided.")

    # Extract a numeric value from one or more possible keys.
    def _val(d, *keys, default=None):
        for key in keys:
            if key in d and d[key] is not None:
                value = d[key]

                if isinstance(value, dict):
                    continue

                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue

        return default

    # ---------------------------------------------------------
    # Determine whether reliable longitudinal data exists.
    #
    # Asymmetry alone is not sufficient. We require at least
    # one measurable longitudinal feature such as diameter,
    # area, border irregularity, or color.
    # ---------------------------------------------------------

    def _has_reliable_longitudinal_measurement(features: dict) -> bool:
        diameter = _val(
            features,
            "diameter_mm",
            "max_diameter_mm",
            "diameter_pixels",
            "max_diameter_pixels",
        )

        area = _val(
            features,
            "area",
            "lesion_area",
        )

        border = _val(
            features,
            "border_irregularity",
            "border_irregularity_score",
        )

        color = _val(
            features,
            "color_index",
        )

        if color is None and isinstance(features.get("color"), dict):
            color = _val(features["color"], "index")

        return any(
            value is not None
            for value in (diameter, area, border, color)
        )

    if not _has_reliable_longitudinal_measurement(previous_features):
        return _unavailable(
            "Reliable longitudinal data unavailable for previous evaluation."
        )

    if not _has_reliable_longitudinal_measurement(current_features):
        return _unavailable(
            "Reliable longitudinal data unavailable for current evaluation."
        )

    # ---------------------------------------------------------
    # Diameter
    # ---------------------------------------------------------

    prev_d = _val(
        previous_features,
        "diameter_mm",
        "max_diameter_mm",
        "diameter_pixels",
        "max_diameter_pixels",
    )

    curr_d = _val(
        current_features,
        "diameter_mm",
        "max_diameter_mm",
        "diameter_pixels",
        "max_diameter_pixels",
    )

    diameter_change = (
        round(curr_d - prev_d, 2)
        if prev_d is not None and curr_d is not None
        else 0.0
    )

    # ---------------------------------------------------------
    # Area
    # ---------------------------------------------------------

    prev_area = _val(
        previous_features,
        "area",
        "lesion_area",
    )

    curr_area = _val(
        current_features,
        "area",
        "lesion_area",
    )

    area_change = (
        round(curr_area - prev_area, 2)
        if prev_area is not None and curr_area is not None
        else 0.0
    )

    # ---------------------------------------------------------
    # Asymmetry
    # ---------------------------------------------------------

    prev_asym = _val(
        previous_features,
        "asymmetry",
        "asymmetry_score",
    )

    curr_asym = _val(
        current_features,
        "asymmetry",
        "asymmetry_score",
    )

    if prev_asym is None:
        prev_asym = 0.0

    if curr_asym is None:
        curr_asym = 0.0

    asymmetry_change = round(curr_asym - prev_asym, 4)

    # ---------------------------------------------------------
    # Border
    # ---------------------------------------------------------

    prev_border = _val(
        previous_features,
        "border_irregularity",
        "border_irregularity_score",
    )

    curr_border = _val(
        current_features,
        "border_irregularity",
        "border_irregularity_score",
    )

    if prev_border is None:
        prev_border = 0.0

    if curr_border is None:
        curr_border = 0.0

    border_change = round(curr_border - prev_border, 4)

    # ---------------------------------------------------------
    # Color
    # ---------------------------------------------------------

    prev_color = _val(
        previous_features,
        "color_index",
    )

    if prev_color is None and isinstance(previous_features.get("color"), dict):
        prev_color = _val(previous_features["color"], "index")

    curr_color = _val(
        current_features,
        "color_index",
    )

    if curr_color is None and isinstance(current_features.get("color"), dict):
        curr_color = _val(current_features["color"], "index")

    if prev_color is None:
        prev_color = 0.0

    if curr_color is None:
        curr_color = 0.0

    color_change = round(curr_color - prev_color, 4)

    # ---------------------------------------------------------
    # Overall change magnitude
    # ---------------------------------------------------------

    normalized_diameter_change = (
        min(1.0, abs(diameter_change) / 5.0)
        if diameter_change is not None
        else 0.0
    )

    overall_change = round(
        float(
            np.sqrt(
                asymmetry_change**2
                + border_change**2
                + color_change**2
                + normalized_diameter_change**2
            )
        ),
        4,
    )

    # ---------------------------------------------------------
    # Significant change thresholds
    # ---------------------------------------------------------

    change_detected = bool(
        abs(asymmetry_change) > 0.05
        or abs(border_change) > 0.05
        or abs(color_change) > 0.05
        or abs(diameter_change) > 0.5
        or abs(area_change) > 0
        or overall_change > 0.10
    )

    changes = {
        "diameter": {
            "previous": prev_d,
            "current": curr_d,
            "change": diameter_change,
        },
        "area": {
            "previous": prev_area,
            "current": curr_area,
            "change": area_change,
        },
        "asymmetry": {
            "previous": prev_asym,
            "current": curr_asym,
            "change": asymmetry_change,
        },
        "border": {
            "previous": prev_border,
            "current": curr_border,
            "change": border_change,
        },
        "color": {
            "previous": prev_color,
            "current": curr_color,
            "change": color_change,
        },
    }

    return {
        "available": True,
        "change_detected": change_detected,
        "diameter_change": diameter_change,
        "area_change": area_change,
        "asymmetry_change": asymmetry_change,
        "border_change": border_change,
        "color_change": color_change,
        "overall_change": overall_change,
        "label": "Observed temporal image change",
        "description": (
            "Temporal image change observed between previous and current "
            "evaluation. Image changes do not automatically indicate "
            "melanoma progression and require professional dermatological "
            "correlation."
        ),
        "reason": "Reliable longitudinal data available for comparison.",
        "changes": changes,
    }