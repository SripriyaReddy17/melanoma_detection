"""
Specialized Feature Agents for ABCDE Analysis (Rule 28).
Each agent receives only relevant structured findings from AnalysisState.
Agents use deterministic clinical logic and NEVER invent symptoms or patient history.
"""

from typing import Dict, Any


class AsymmetryAgent:
    """A_AGENT: Evaluates geometric and structural asymmetry."""
    def evaluate(self, asym_data: Dict[str, Any], classification_context: Dict[str, Any]) -> Dict[str, Any]:
        score = float(asym_data.get("score", 0.0) if isinstance(asym_data, dict) else asym_data or 0.0)
        severity = asym_data.get("severity", "high" if score >= 0.55 else ("medium" if score >= 0.30 else "low"))
        horiz = asym_data.get("horizontal_asymmetry", 0.0) if isinstance(asym_data, dict) else 0.0
        vert = asym_data.get("vertical_asymmetry", 0.0) if isinstance(asym_data, dict) else 0.0

        if score >= 0.55:
            finding = f"Marked structural asymmetry (score: {score:.2f}, horizontal: {horiz:.2f}, vertical: {vert:.2f})."
            interpretation = "The two halves of the lesion show substantial geometric mismatch, characteristic of asymmetrical radial growth."
        elif score >= 0.30:
            finding = f"Moderate lesion asymmetry (score: {score:.2f})."
            interpretation = "Mild to moderate deviation from bilateral symmetry observed across principal axes."
        else:
            finding = f"Predominantly symmetric contour (score: {score:.2f})."
            interpretation = "Lesion halves demonstrate high degree of mutual reflection and balanced geometry."

        return {
            "feature": "Asymmetry (A)",
            "score": round(score, 4),
            "severity": severity,
            "finding": finding,
            "interpretation": interpretation,
            "method": "mask-based mirror comparison",
        }


class BorderAgent:
    """B_AGENT: Evaluates contour regularity, compactness, and curvature."""
    def evaluate(self, border_data: Dict[str, Any], classification_context: Dict[str, Any]) -> Dict[str, Any]:
        score = float(border_data.get("border_irregularity_score", 0.0) if isinstance(border_data, dict) else border_data or 0.0)
        circ = float(border_data.get("circularity", 1.0) if isinstance(border_data, dict) else 1.0)
        curv_var = float(border_data.get("curvature_variation", 0.0) if isinstance(border_data, dict) else 0.0)

        if score >= 0.50:
            finding = f"Prominent border irregularity (score: {score:.2f}, circularity: {circ:.2f}, curvature var: {curv_var:.2f})."
            interpretation = "Lesion margins exhibit scalloped, jagged contours and abrupt transitions characteristic of irregular peripheral growth."
        elif score >= 0.30:
            finding = f"Mildly irregular or wavy margins (score: {score:.2f})."
            interpretation = "Some border unevenness detected, though without acute angular notches."
        else:
            finding = f"Smooth and well-circumscribed boundary (score: {score:.2f}, circularity: {circ:.2f})."
            interpretation = "Lesion contour is regular, intact, and well-demarcated from surrounding tissue."

        return {
            "feature": "Border (B)",
            "score": round(score, 4),
            "finding": finding,
            "interpretation": interpretation,
            "circularity": round(circ, 4),
            "curvature_variation": round(curv_var, 4),
        }


class ColorAgent:
    """C_AGENT: Evaluates chromatic distribution and HSV variegation."""
    def evaluate(self, color_data: Dict[str, Any], classification_context: Dict[str, Any]) -> Dict[str, Any]:
        h_std = float(color_data.get("hue_variation", color_data.get("h_raw", 0.0)))
        s_std = float(color_data.get("saturation_variation", color_data.get("s_raw", 0.0)))
        v_std = float(color_data.get("brightness_variation", color_data.get("v_raw", 0.0)))
        diversity = float(color_data.get("color_diversity", color_data.get("index", 0.0)))
        dominant = color_data.get("dominant_colors", [])

        if diversity >= 0.50 or h_std > 20.0:
            finding = f"Marked color variation and pigment variegation (diversity: {diversity:.2f}, σH: {h_std:.1f}, σS: {s_std:.1f}, σV: {v_std:.1f})."
            interpretation = "Multiple distinct color shades and heterogeneous pigment clustering present inside the lesion."
        elif diversity >= 0.30:
            finding = f"Moderate color diversity (index: {diversity:.2f})."
            interpretation = "Two to three harmonized pigment tones detected across the lesion surface."
        else:
            finding = f"Uniform, homogeneous pigmentation (diversity: {diversity:.2f})."
            interpretation = "Coloration is uniform with minimal chromatic standard deviation across HSV channels."

        return {
            "feature": "Color (C)",
            "diversity_index": round(diversity, 4),
            "hue_variation": round(h_std, 2),
            "saturation_variation": round(s_std, 2),
            "brightness_variation": round(v_std, 2),
            "finding": finding,
            "interpretation": interpretation,
            "dominant_colors": dominant,
        }


class DiameterAgent:
    """D_AGENT: Evaluates lesion extent and verified physical millimeter calibration."""
    def evaluate(self, dia_data: Dict[str, Any], classification_context: Dict[str, Any]) -> Dict[str, Any]:
        d_px = dia_data.get("diameter_pixels") or dia_data.get("max_diameter_pixels", 0.0)
        d_mm = dia_data.get("diameter_mm") or dia_data.get("max_diameter_mm")
        cal_used = bool(dia_data.get("calibration_used") or dia_data.get("available", False))

        if cal_used and d_mm is not None:
            exceeds = float(d_mm) > 6.0
            finding = f"Calibrated maximum diameter: {float(d_mm):.1f} mm ({'exceeds' if exceeds else 'within'} the 6 mm ABCDE reference threshold)."
            interpretation = (
                f"Physical diameter is {'larger than' if exceeds else 'smaller than'} the traditional 6 mm screening guideline, "
                "established via verified physical calibration marker/ruler."
            )
        else:
            finding = f"Uncalibrated extent: approximately {float(d_px):.0f} pixels."
            interpretation = "Physical diameter in millimeters is unavailable because no valid calibration marker or ruler was detected. Physical size cannot be clinically certified without calibration."

        return {
            "feature": "Diameter (D)",
            "available": bool(cal_used and d_mm is not None),
            "diameter_pixels": round(float(d_px), 1) if d_px else 0.0,
            "diameter_mm": round(float(d_mm), 2) if (cal_used and d_mm is not None) else None,
            "calibration_used": cal_used,
            "measurement_method": "maximum lesion extent via convex hull / fitted ellipse",
            "finding": finding,
            "interpretation": interpretation,
        }


class EvolutionAgent:
    """E_AGENT: Evaluates longitudinal temporal image changes across visits."""
    def evaluate(self, evo_data: Dict[str, Any], classification_context: Dict[str, Any]) -> Dict[str, Any]:
        available = bool(evo_data and evo_data.get("available", False))
        if not available:
            return {
                "feature": "Evolution (E)",
                "available": False,
                "reason": "Previous image not provided",
                "finding": "Longitudinal comparison unavailable.",
                "interpretation": "Evolution cannot be assessed from a single image. A prior baseline dermoscopic image is required to detect temporal changes.",
            }

        chg_detected = evo_data.get("change_detected", False)
        chg_mag = float(evo_data.get("overall_change", 0.0))
        d_change = evo_data.get("diameter_change", 0.0)
        area_change = evo_data.get("area_change", 0.0)

        if chg_detected:
            finding = f"Observed temporal image change: change magnitude {chg_mag:.2f} (diameter delta: {d_change}, area delta: {area_change})."
            interpretation = "Documented temporal changes in lesion geometry or coloration observed over time. Does not automatically imply malignancy but indicates an evolving lesion."
        else:
            finding = f"Temporal lesion stability observed (change magnitude: {chg_mag:.2f})."
            interpretation = "No significant morphologic, chromatic, or dimensional shift detected compared to the prior examination."

        return {
            "feature": "Evolution (E)",
            "available": True,
            "change_detected": chg_detected,
            "overall_change": round(chg_mag, 4),
            "diameter_change": d_change,
            "area_change": area_change,
            "asymmetry_change": evo_data.get("asymmetry_change", 0.0),
            "border_change": evo_data.get("border_change", 0.0),
            "color_change": evo_data.get("color_change", 0.0),
            "label": "Observed temporal image change",
            "finding": finding,
            "interpretation": interpretation,
        }


# Agent instances
A_AGENT = AsymmetryAgent()
B_AGENT = BorderAgent()
C_AGENT = ColorAgent()
D_AGENT = DiameterAgent()
E_AGENT = EvolutionAgent()
