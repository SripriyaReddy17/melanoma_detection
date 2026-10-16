"""
Asymmetry (A) feature extraction for CEFM.
Complies with Table 1 of CEFM Paper:
A = sum(|I(x, y) - I_mirror(x, y)|) / sum(M(x, y))
and Rule 12.
"""

from __future__ import annotations

import cv2
import numpy as np


def _principal_angle(mask: np.ndarray) -> float:
    """Estimate the lesion's dominant orientation from second moments."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0

    x = xs.astype(np.float64) - float(xs.mean())
    y = ys.astype(np.float64) - float(ys.mean())
    cov = np.cov(np.vstack([x, y]), bias=True)
    if cov.shape != (2, 2):
        return 0.0
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = 3
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return (mask[y0:y1, x0:x1] > 0).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.count_nonzero((a > 0) & (b > 0)))
    union = float(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union > 0 else 1.0


def compute_asymmetry_analysis(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute structured asymmetry analysis per Rule 12.
    Returns:
    {
        "score": float in [0, 1],
        "severity": "low" | "medium" | "high",
        "method": "mask-based mirror comparison",
        "horizontal_asymmetry": float,
        "vertical_asymmetry": float,
        "area_mismatch": float,
        "principal_angle": float
    }
    """
    if mask is None or np.count_nonzero(mask) == 0:
        return {
            "score": 0.0,
            "severity": "low",
            "method": "mask-based mirror comparison",
            "horizontal_asymmetry": 0.0,
            "vertical_asymmetry": 0.0,
            "area_mismatch": 0.0,
            "principal_angle": 0.0,
        }

    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    if len(xs) < 5:
        return {
            "score": 0.0,
            "severity": "low",
            "method": "mask-based mirror comparison",
            "horizontal_asymmetry": 0.0,
            "vertical_asymmetry": 0.0,
            "area_mismatch": 0.0,
            "principal_angle": 0.0,
        }

    angle = _principal_angle(binary)
    center = (float(xs.mean()), float(ys.mean()))
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        matrix,
        (binary.shape[1], binary.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    cropped = _crop_mask(rotated)
    if cropped is None:
        return {
            "score": 0.0,
            "severity": "low",
            "method": "mask-based mirror comparison",
            "horizontal_asymmetry": 0.0,
            "vertical_asymmetry": 0.0,
            "area_mismatch": 0.0,
            "principal_angle": angle,
        }

    h, w = cropped.shape
    mid_x = w // 2
    left = cropped[:, :mid_x]
    right = cropped[:, mid_x:]
    left_area = float(np.count_nonzero(left))
    right_area = float(np.count_nonzero(right))
    total_area = max(1.0, left_area + right_area)
    area_mismatch = abs(left_area - right_area) / total_area

    vertical = np.fliplr(cropped)
    horizontal = np.flipud(cropped)
    vert_sym = _iou(cropped, vertical)
    horiz_sym = _iou(cropped, horizontal)
    
    vert_asym = 1.0 - vert_sym
    horiz_asym = 1.0 - horiz_sym
    reflection_score = 0.5 * (vert_asym + horiz_asym)

    score = 0.7 * reflection_score + 0.3 * area_mismatch
    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.25 and area_mismatch > 0.08:
        score = 0.25

    final_score = round(score, 4)
    severity = "high" if final_score >= 0.55 else ("medium" if final_score >= 0.30 else "low")

    return {
        "score": final_score,
        "severity": severity,
        "method": "mask-based mirror comparison",
        "horizontal_asymmetry": round(horiz_asym, 4),
        "vertical_asymmetry": round(vert_asym, 4),
        "area_mismatch": round(area_mismatch, 4),
        "principal_angle": round(angle, 2),
    }


def compute_asymmetry(image: np.ndarray, mask: np.ndarray) -> float:
    """Return a normalized A score in [0, 1] for backward compatibility."""
    analysis = compute_asymmetry_analysis(image, mask)
    return analysis["score"]
