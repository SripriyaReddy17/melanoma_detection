"""
ABCDE Vector representation for CEFM Cross-modal Alignment (Rule 17).
Encodes A, B, C, D, E into a standardized feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np
import torch


@dataclass
class ABCDEVector:
    """Unified container for extracted ABCDE clinical features."""
    asymmetry_score: float = 0.0
    horizontal_asymmetry: float = 0.0
    vertical_asymmetry: float = 0.0
    border_irregularity: float = 0.0
    circularity: float = 1.0
    radial_distance_variation: float = 0.0
    curvature_variation: float = 0.0
    color_h_norm: float = 0.0
    color_s_norm: float = 0.0
    color_v_norm: float = 0.0
    color_diversity: float = 0.0
    diameter_mm_norm: float = 0.0
    calibration_available: float = 0.0
    evolution_change_flag: float = 0.0
    evolution_magnitude: float = 0.0
    evolution_available: float = 0.0

    def to_list(self) -> list[float]:
        return [
            float(self.asymmetry_score),
            float(self.horizontal_asymmetry),
            float(self.vertical_asymmetry),
            float(self.border_irregularity),
            float(self.circularity),
            float(self.radial_distance_variation),
            float(self.curvature_variation),
            float(self.color_h_norm),
            float(self.color_s_norm),
            float(self.color_v_norm),
            float(self.color_diversity),
            float(self.diameter_mm_norm),
            float(self.calibration_available),
            float(self.evolution_change_flag),
            float(self.evolution_magnitude),
            float(self.evolution_available),
        ]

    def to_numpy(self) -> np.ndarray:
        return np.array(self.to_list(), dtype=np.float32)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.to_list(), dtype=torch.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_abcde_vector(
    abcde: dict[str, Any],
    evolution: Optional[dict[str, Any]] = None,
) -> ABCDEVector:
    """
    Construct an ABCDEVector from extracted features and optional evolution dictionary.
    Handles missing calibration or evolution explicitly without inventing data.
    """
    asym = abcde.get("asymmetry", 0.0)
    asym_score = float(asym if not isinstance(asym, dict) else asym.get("score", 0.0))
    asym_h = float(asym.get("horizontal_asymmetry", 0.0) if isinstance(asym, dict) else 0.0)
    asym_v = float(asym.get("vertical_asymmetry", 0.0) if isinstance(asym, dict) else 0.0)

    border = abcde.get("border_irregularity", 0.0)
    border_score = float(border if not isinstance(border, dict) else border.get("border_irregularity_score", 0.0))
    circ = float(border.get("circularity", 1.0) if isinstance(border, dict) else 1.0)
    rad_var = float(border.get("radial_distance_variation", 0.0) if isinstance(border, dict) else 0.0)
    curv_var = float(border.get("curvature_variation", 0.0) if isinstance(border, dict) else 0.0)

    color = abcde.get("color", {}) if isinstance(abcde.get("color"), dict) else {}
    h_norm = float(color.get("h", 0.0))
    s_norm = float(color.get("s", 0.0))
    v_norm = float(color.get("v", 0.0))
    c_div = float(color.get("color_diversity", color.get("index", 0.0)))

    dia = abcde.get("diameter", {}) if isinstance(abcde.get("diameter"), dict) else {}
    cal_used = bool(dia.get("calibration_used") or dia.get("available", False))
    d_mm = dia.get("diameter_mm") or dia.get("max_diameter_mm")
    d_mm_norm = min(1.0, float(d_mm) / 20.0) if (cal_used and d_mm is not None) else 0.0

    evo = evolution or {}
    evo_avail = bool(evo.get("available", False))
    evo_change = 1.0 if (evo_avail and evo.get("change_detected", False)) else 0.0
    evo_mag = min(1.0, float(evo.get("overall_change", 0.0))) if evo_avail else 0.0

    return ABCDEVector(
        asymmetry_score=asym_score,
        horizontal_asymmetry=asym_h,
        vertical_asymmetry=asym_v,
        border_irregularity=border_score,
        circularity=circ,
        radial_distance_variation=rad_var,
        curvature_variation=curv_var,
        color_h_norm=h_norm,
        color_s_norm=s_norm,
        color_v_norm=v_norm,
        color_diversity=c_div,
        diameter_mm_norm=d_mm_norm,
        calibration_available=1.0 if cal_used else 0.0,
        evolution_change_flag=evo_change,
        evolution_magnitude=evo_mag,
        evolution_available=1.0 if evo_avail else 0.0,
    )
