"""Physical lesion diameter estimation for ABCDE D."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external lesion contour."""
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _major_axis_length(contour: np.ndarray) -> float:
    """Estimate lesion major axis length using a fitted ellipse."""
    if contour is None or len(contour) < 5:
        return 0.0

    try:
        contour_approx = cv2.approxPolyDP(
            contour,
            0.01 * cv2.arcLength(contour, True),
            True,
        )
        if len(contour_approx) < 5:
            return 0.0

        (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour_approx)
        return round(float(max(major_axis, minor_axis)), 2)
    except cv2.error:
        return 0.0


def compute_diameter(mask: np.ndarray) -> float:
    """Major lesion diameter in pixels, using a stable contour-based estimate."""
    if mask is None or np.count_nonzero(mask) == 0:
        return 0.0

    contour = _largest_contour(mask)
    if contour is None or len(contour) < 2:
        return 0.0

    diameter = _major_axis_length(contour)
    if diameter > 0:
        return diameter

    hull = cv2.convexHull(contour)
    points = hull.reshape(-1, 2).astype(np.float32)
    if len(points) < 2:
        return 0.0

    max_distance = 0.0
    for i in range(len(points)):
        diffs = points[i + 1:] - points[i]
        if len(diffs) == 0:
            continue
        distances = np.sqrt((diffs * diffs).sum(axis=1))
        max_distance = max(max_distance, float(distances.max()))
    return round(max_distance, 2)


def compute_min_diameter(mask: np.ndarray) -> float:
    """Approximate minor lesion diameter in pixels."""
    if mask is None or np.count_nonzero(mask) == 0:
        return 0.0

    contour = _largest_contour(mask)
    if contour is None:
        return 0.0

    if len(contour) < 5:
        area = cv2.contourArea(contour)
        return round(
            2.0 * np.sqrt(area / np.pi),
            2,
        ) if area > 0 else 0.0

    (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
    return round(float(min(axis_a, axis_b)), 2)


def compute_diameter_mm(
    mask: np.ndarray,
    pixels_per_mm: Optional[float] = None,
) -> dict:
    """Return pixel measurements and calibrated mm values only when valid calibration exists."""
    max_px = compute_diameter(mask)
    min_px = compute_min_diameter(mask)

    ppm = (
        float(pixels_per_mm)
        if pixels_per_mm is not None
        else None
    )

    result = {
        "available": False,
        "calibration_used": False,
        "diameter_pixels": max_px,
        "diameter_mm": None,
        "method": "maximum lesion extent via convex hull / fitted ellipse",
        "reason": (
            "Diameter in mm is not available because no valid physical calibration "
            "reference was provided for this image. Pixel-based diameter may still "
            "be computed, but it is not clinically calibrated."
        ),
        "max_diameter_pixels": max_px,
        "min_diameter_pixels": min_px,
        "pixels_per_mm": ppm,
        "max_diameter_mm": None,
        "min_diameter_mm": None,
        "exceeds_6mm": False,
    }

    if ppm is not None and ppm > 0:
        max_mm = max_px / ppm
        min_mm = min_px / ppm
        result["available"] = True
        result["calibration_used"] = True
        result["diameter_mm"] = round(max_mm, 2)
        result["reason"] = (
            "Diameter in mm was computed from a valid physical calibration reference."
        )
        result["max_diameter_mm"] = round(max_mm, 2)
        result["min_diameter_mm"] = round(min_mm, 2)
        result["exceeds_6mm"] = bool(max_mm > 6.0)

    return result
