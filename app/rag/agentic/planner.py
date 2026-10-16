"""
Query Planning Agent for Agentic RAG (Rule 27).
Analyzes current pipeline findings and dynamically generates targeted clinical retrieval queries.
"""

from typing import List, Dict, Any
from app.rag.agentic.state import AnalysisState


class QueryPlanningAgent:
    """Formulates clinical retrieval queries grounded strictly in AnalysisState data."""

    def plan_queries(self, state: AnalysisState) -> List[Dict[str, Any]]:
        queries = []

        classification = state.classification or {}
        abcde = state.abcde or {}
        evolution = state.abcde.get("evolution") or state.report.get("evolution") or {}
        clip = state.clip or {}

        pred = classification.get("predicted_class", "unknown")
        mel_prob = classification.get("melanoma_probability", 0.0)

        # 1. Classification & Overview query
        queries.append({
            "target": "classification_overview",
            "feature": "overview",
            "query": f"Clinical significance and diagnostic guidelines for dermoscopic assessment of {pred} lesions with probability {mel_prob:.2f}.",
            "priority": "high",
        })

        # 2. Asymmetry (A) query
        asym = abcde.get("asymmetry", 0.0)
        asym_score = asym if not isinstance(asym, dict) else asym.get("score", 0.0)
        queries.append({
            "target": "asymmetry",
            "feature": "A",
            "query": f"Dermoscopic significance of lesion asymmetry score {asym_score:.2f} and mirror geometric imbalance.",
            "priority": "high" if asym_score > 0.4 else "medium",
        })

        # 3. Border (B) query
        border = abcde.get("border_irregularity", 0.0)
        border_score = border if not isinstance(border, dict) else border.get("border_irregularity_score", 0.0)
        queries.append({
            "target": "border",
            "feature": "B",
            "query": f"Border irregularity, notch formation, and contour curvature variation score {border_score:.2f} in melanoma screening.",
            "priority": "high" if border_score > 0.4 else "medium",
        })

        # 4. Color (C) query
        color = abcde.get("color", {}) if isinstance(abcde.get("color"), dict) else {}
        c_index = color.get("index", color.get("color_diversity", 0.0))
        queries.append({
            "target": "color",
            "feature": "C",
            "query": f"Color variegation, multi-chromatic pigment patterns, and HSV color standard deviation index {c_index:.2f}.",
            "priority": "high" if c_index > 0.4 else "medium",
        })

        # 5. Diameter (D) query
        dia = abcde.get("diameter", {}) if isinstance(abcde.get("diameter"), dict) else {}
        cal_used = dia.get("calibration_used", False)
        d_mm = dia.get("diameter_mm") or dia.get("max_diameter_mm")
        if cal_used and d_mm is not None:
            dia_query = f"Physical lesion diameter {d_mm:.1f} mm relative to clinical 6 mm threshold with verified calibration."
        else:
            dia_query = "Dermoscopic diameter measurement and physical calibration requirements for skin lesions."
        queries.append({
            "target": "diameter",
            "feature": "D",
            "query": dia_query,
            "priority": "medium",
        })

        # 6. Evolution (E) query
        evo_avail = evolution.get("available", False)
        if evo_avail:
            chg = evolution.get("change_detected", False)
            evo_query = f"Longitudinal lesion evolution and temporal change detection ({'change observed' if chg else 'stable'})."
        else:
            evo_query = "Role of longitudinal temporal evolution and requirements for comparison in dermoscopy."
        queries.append({
            "target": "evolution",
            "feature": "E",
            "query": evo_query,
            "priority": "high" if evo_avail else "low",
        })

        # 7. Model limitations & Safety query
        queries.append({
            "target": "limitations_safety",
            "feature": "safety",
            "query": "AI dermoscopy model limitations, artifact sensitivity, and recommendations for professional dermatology review.",
            "priority": "high",
        })

        return queries
