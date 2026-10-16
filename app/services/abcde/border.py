"""
Border Irregularity (B) feature extraction for CEFM.
Complies with Table 1 of CEFM Paper:
B2 = (1/N) * sum(kappa_i), where kappa_i = delta_theta_i / delta_s_i
and Rule 13 descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_border_analysis(mask: np.ndarray) -> dict:
    """
    Extract quantitative border irregularity descriptors per Rule 13.
    Returns:
    {
        "border_irregularity_score": float in [0, 1],
        "perimeter": float,
        "area": float,
        "circularity": float,
        "compactness": float,
        "radial_distance_variation": float,
        "curvature_variation": float,
        "mean_curvature": float
    }
    """
    default_res = {
        "border_irregularity_score": 0.0,
        "perimeter": 0.0,
        "area": 0.0,
        "circularity": 1.0,
        "compactness": 1.0,
        "radial_distance_variation": 0.0,
        "curvature_variation": 0.0,
        "mean_curvature": 0.0,
    }

    if mask is None or np.count_nonzero(mask) == 0:
        return default_res

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return default_res

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 0.0 or perimeter <= 0.0:
        return default_res

    # 1. Circularity: 4*pi*area / perimeter^2 (1.0 for perfect circle, lower for irregular)
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    circularity = float(np.clip(circularity, 0.0, 1.0))

    # 2. Compactness: perimeter^2 / (4*pi*area)
    compactness = 1.0 / max(circularity, 1e-5)

    # 3. Radial distance variation
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        pts = contour.reshape(-1, 2)
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

    pts = contour.reshape(-1, 2).astype(np.float64)
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    mean_radius = float(np.mean(radii)) if len(radii) > 0 else 1.0
    radial_std = float(np.std(radii)) if len(radii) > 0 else 0.0
    radial_distance_variation = radial_std / max(mean_radius, 1e-5)

    # 4. Local curvature calculation (Table 1: kappa_i = delta_theta / delta_s)
    curvatures = []
    n_pts = len(pts)
    step = max(1, n_pts // 100)  # sample ~100 points
    sampled_indices = list(range(0, n_pts, step))
    
    for idx, i in enumerate(sampled_indices):
        p_prev = pts[sampled_indices[(idx - 1) % len(sampled_indices)]]
        p_curr = pts[i]
        p_next = pts[sampled_indices[(idx + 1) % len(sampled_indices)]]
        
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        
        ds1 = np.linalg.norm(v1)
        ds2 = np.linalg.norm(v2)
        ds = 0.5 * (ds1 + ds2)
        
        if ds > 1e-5:
            theta1 = np.arctan2(v1[1], v1[0])
            theta2 = np.arctan2(v2[1], v2[0])
            dtheta = abs((theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi)
            curvatures.append(dtheta / ds)

    if curvatures:
        mean_curvature = float(np.mean(curvatures))
        curvature_variation = float(np.std(curvatures))
    else:
        mean_curvature = 0.0
        curvature_variation = 0.0

    # Composite border irregularity score in [0, 1]
    irregularity = 0.5 * (1.0 - circularity) + 0.3 * np.clip(radial_distance_variation, 0.0, 1.0) + 0.2 * np.clip(curvature_variation * 10.0, 0.0, 1.0)
    score = float(np.clip(irregularity, 0.0, 1.0))

    return {
        "border_irregularity_score": round(score, 4),
        "perimeter": round(perimeter, 2),
        "area": round(area, 2),
        "circularity": round(circularity, 4),
        "compactness": round(compactness, 4),
        "radial_distance_variation": round(radial_distance_variation, 4),
        "curvature_variation": round(curvature_variation, 4),
        "mean_curvature": round(mean_curvature, 4),
    }


def compute_border_irregularity(mask: np.ndarray) -> float:
    """Return a normalized B score in [0, 1] for backward compatibility."""
    analysis = compute_border_analysis(mask)
    return analysis["border_irregularity_score"]
