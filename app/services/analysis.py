"""Full CEFM analysis pipeline with normalized ABCD features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.services.image_validation import (
    validate_image,
    load_image,
    check_post_segmentation_quality,
)

from app.services.calibration import estimate_pixels_per_mm
from app.services.image_type import image_type_detector
from app.services.segmentation import segment_lesion

from app.services.abcde.asymmetry import compute_asymmetry
from app.services.abcde.border import compute_border_irregularity
from app.services.abcde.color import compute_color_variation
from app.services.abcde.composite import compute_abcde_score
from app.services.abcde.diameter import compute_diameter_mm
from app.services.abcde.evolution import compute_evolution

from app.services.classification import classify
from app.services.report_generation import generate_structured_report
from app.services.report_fact_checker import validate_report


SEGMENTATION_MODEL_NAME = "UltraLight VM-UNet"


# ============================================================
# RAG IMPORT
# ============================================================

from app.rag.rag_service import retrieve_melanoma_evidence
from app.rag.agentic.clinical_agent import ClinicalRAGAgent


# ============================================================
# SEGMENTATION OVERLAY HELPER
# ============================================================

def _save_segmentation_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
) -> None:
    """
    Save a coloured mask overlay (red fill + cyan contour)
    as a PNG next to the uploaded image file.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Semi-transparent red fill over the lesion
    lesion_pixels = mask == 1
    overlay[lesion_pixels] = (
        0.35 * overlay[lesion_pixels]
        + 0.65 * np.array([0, 0, 220], dtype=np.float32)
    ).astype(np.uint8)

    # Cyan contour outline
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    cv2.imwrite(str(save_path), overlay)


# ============================================================
# ABCDE FEATURE EXTRACTION
# ============================================================

def extract_abcde_features(
    image: np.ndarray,
    mask: np.ndarray,
    pixels_per_mm: Optional[float] = None,
) -> dict:
    """Compute normalized A/B/C features and calibrated D."""

    diameter = compute_diameter_mm(
        mask,
        pixels_per_mm=pixels_per_mm,
    )

    color = compute_color_variation(
        image,
        mask,
    )

    return {
        "asymmetry": round(
            float(
                compute_asymmetry(
                    image,
                    mask,
                )
            ),
            4,
        ),

        "border_irregularity": round(
            float(
                compute_border_irregularity(
                    mask
                )
            ),
            4,
        ),

        "color": color,

        "diameter_pixels": round(
            float(
                diameter[
                    "max_diameter_pixels"
                ]
            ),
            2,
        ),

        "diameter": diameter,

        "diameter_mm_available": diameter.get(
            "available",
            False,
        ),
    }


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_image(
    image_path: str,
    original_filename: str,
    previous_features: Optional[dict] = None,
    remove_hair_flag: bool = False,
    pixels_per_mm: Optional[float] = None,
) -> dict:
    """
    Run validation, calibration, segmentation,
    ABCDE, classification, RAG, evolution and report.
    """

    # ========================================================
    # 1. Basic image validation
    # ========================================================

    validation = validate_image(
        image_path,
        original_filename,
    )

    if not validation["valid"]:
        return {
            "status": "invalid",
            "validation": validation,
            "segmentation": {
                "model": SEGMENTATION_MODEL_NAME,
                "status": "not_run",
                "mask_available": False,
            },
            "segmentation_path": None,
            "rejection_stage": "basic_validation",
        }

    image = load_image(
        image_path
    )

    # ========================================================
    # 2. Optional hair removal
    # ========================================================

    if remove_hair_flag:

        from app.services.segmentation import remove_hair

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        image = cv2.cvtColor(
            remove_hair(image_bgr),
            cv2.COLOR_BGR2RGB,
        )

    _seg_overlay_path = (
        Path(image_path).parent
        / (Path(image_path).stem + "_seg.png")
    )
    segmentation_path: Optional[str] = None

    # ========================================================
    # 3. Physical calibration
    # ========================================================

    # A valid physical reference is mandatory
    # for patient-image D.
    calibration = estimate_pixels_per_mm(
        image,
        lesion_mask=None,
    )

    validation["calibration"] = calibration

    if not calibration.get("valid"):

        validation["calibration_rejection"] = {
            "valid": False,
            "message": calibration.get(
                "message",
                "No valid calibration reference was detected.",
            ),
        }

        return {
            "status": "invalid",
            "validation": validation,
            "calibration": calibration,
            "segmentation": {
                "model": SEGMENTATION_MODEL_NAME,
                "status": "not_run",
                "mask_available": False,
            },
            "segmentation_path": segmentation_path,
            "rejection_stage": "calibration",
        }

    # ========================================================
    # 4. Segmentation
    # ========================================================

    # The calibration module may return either:
    #   corners        -> square marker
    #   exclude_polygon -> ruler strip
    #
    # Use whichever exclusion geometry is available.

    exclude_polygon = (
        calibration.get("exclude_polygon")
        or calibration.get("corners")
    )

    mask = segment_lesion(
        image_path,
        remove_hair_flag=remove_hair_flag,
        exclude_polygon=exclude_polygon,
    )
    segmentation = {
        "model": SEGMENTATION_MODEL_NAME,
        "status": "completed",
        "mask_available": True,
    }
    _save_segmentation_overlay(
        image,
        mask,
        _seg_overlay_path,
    )
    segmentation_path = str(_seg_overlay_path)

    # ========================================================
    # 6. Post-segmentation quality gate
    # ========================================================

    post_quality = check_post_segmentation_quality(
        image,
        mask,
        calibration,
    )

    validation[
        "post_segmentation_quality"
    ] = post_quality

    if not post_quality["valid"]:

        return {
            "status": "invalid",
            "validation": validation,
            "calibration": calibration,
            "segmentation": segmentation,
            "segmentation_path": segmentation_path,
            "rejection_stage": "post_segmentation_quality",
        }

    # ========================================================
    # 7. Use automatic physical scale
    # ========================================================

    auto_ppm = calibration.get(
        "pixels_per_mm"
    )

    # Patient workflow never relies on a manually typed scale.
    pixels_per_mm = (
        float(auto_ppm)
        if (
            auto_ppm is not None
            and float(auto_ppm) > 0
        )
        else None
    )

    if pixels_per_mm is None:

        validation["calibration_rejection"] = {
            "valid": False,
            "message": (
                "Physical calibration could not "
                "be established."
            ),
        }

        return {
            "status": "invalid",
            "validation": validation,
            "calibration": calibration,
            "segmentation": segmentation,
            "segmentation_path": segmentation_path,
            "rejection_stage": "calibration",
        }

    # ========================================================
    # 8. ABCDE features
    # ========================================================

    features = extract_abcde_features(
        image,
        mask,
        pixels_per_mm=pixels_per_mm,
    )

    color = features["color"]
    diameter = features["diameter"]
    flat = {
        "asymmetry": float(features["asymmetry"]),
        "border_irregularity": float(
            features["border_irregularity"]
        ),

        # ----------------------------------------------------
        # Color variation
        # ----------------------------------------------------
        "color_h": float(color.get("h", 0.0)),
        "color_s": float(color.get("s", 0.0)),
        "color_v": float(color.get("v", 0.0)),
        "color_index": float(color.get("index", 0.0)),

        # ----------------------------------------------------
        # Diameter
        # ----------------------------------------------------
        "diameter_pixels": float(
            features["diameter_pixels"]
        ),

        "diameter_mm": (
            float(diameter.get("max_diameter_mm"))
            if diameter.get("max_diameter_mm") is not None
            else None
        ),

        "diameter_mm_available": bool(
            diameter.get("available", False)
        ),
    }

    # Image type is auxiliary metadata; it does not alter the clinical pipeline.
    try:
        image_type = image_type_detector.predict(image_path)
    except Exception:
        image_type = {"type": "unknown"}

    # ========================================================
    # 5. ViT classification
    # ========================================================

    classification = classify(
        image_path,
        flat,
    )

    # ========================================================
    # 10. Evolution
    # ========================================================

    evolution = compute_evolution(
        previous_features,
        flat,
    )

    flat[
        "evolution_change_detected"
    ] = evolution.get(
        "change_detected",
        False,
    )

    abcde_risk = compute_abcde_score(flat)
    flat["abcde_risk_score"] = abcde_risk["score"]
    flat["abcde_risk_label"] = abcde_risk["label"]
    features["composite_score"] = abcde_risk

    # ========================================================
    # 11. Explanation
    # ========================================================

    explanation = {
        "summary": _explain(
            classification,
            features,
            evolution,
        ),
        "abcde_risk": abcde_risk,
    }

    # ========================================================
    # 12. RAG MEDICAL EVIDENCE (AGENTIC RAG)
    # ========================================================

    rag_evidence = {
        "query": "",
        "documents": [],
    }

    try:
        clinical_agent = ClinicalRAGAgent()
        rag_evidence = clinical_agent.run(
            classification=classification,
            abcde=flat,
            evolution=evolution,
            top_k=3,
        )
        rag_evidence = {
            **rag_evidence,
            "retrieval_mode": "agentic_rag",
            "agentic_rag": True,
        }

    except Exception as agent_exc:

        # Fall back to conventional RAG if agent encounters an issue
        agentic_rag_error = (
            f"Agentic RAG failed ({type(agent_exc).__name__})."
        )
        print(
            "[AGENTIC RAG WARNING] "
            f"{agentic_rag_error} Attempting conventional RAG fallback."
        )

        try:
            rag_evidence = retrieve_melanoma_evidence(
                classification=classification,
                abcde=flat,
                evolution=evolution,
                top_k=3,
            )
            rag_evidence = {
                **rag_evidence,
                "retrieval_mode": "conventional_rag_fallback",
                "agentic_rag": False,
                "agentic_rag_error": agentic_rag_error,
            }
        except Exception as exc:

            # RAG should not destroy the main
            # melanoma analysis if the RAG service
            # is temporarily unavailable.

            retrieval_error = (
                f"Conventional RAG failed ({type(exc).__name__})."
            )
            print(
                f"[RAG WARNING] {retrieval_error}"
            )

            rag_evidence = {
                "query": "",
                "documents": [],
                "retrieval_mode": "retrieval_failed",
                "agentic_rag": False,
                "agentic_rag_error": agentic_rag_error,
                "error": retrieval_error,
            }

    # ========================================================
    # 13. Supplementary multimodal evidence after RAG
    # ========================================================
    from app.services.multimodal import run_multimodal_pipeline
    multimodal_result = run_multimodal_pipeline(
        classification=classification,
        abcde=features,
        evolution=evolution,
        image_path=image_path,
        segmentation_result=segmentation,
    )
    contrastive_alignment = multimodal_result.get(
        "contrastive_alignment",
        {},
    )

    # ========================================================
    # 14. Structured report
    # ========================================================

    report = generate_structured_report(
        validation=validation,
        classification=classification,
        abcde=flat,
        evolution=evolution,
        explanation=explanation,
        rag_evidence=rag_evidence,
    )

    # ========================================================
    # 14A. Report fact-checking
    # ========================================================

    fact_check = validate_report(
        report=report,
        classification=classification,
        abcde=flat,
        evolution=evolution,
        rag_evidence=rag_evidence,
    )

    report["fact_check"] = fact_check

    # ========================================================
    # 15. Final result
    # ========================================================

    return {
        "status": "success",

        "validation": validation,

        "image_type": image_type,

        "calibration": calibration,

        "segmentation": segmentation,

        "segmentation_path": segmentation_path,

        "features": flat,

        "abcde": features,

        "abcde_risk": abcde_risk,

        "classification": classification,

        "evolution": evolution,

        "multimodal": multimodal_result,

        "contrastive_alignment": contrastive_alignment,

        "explanation": explanation,

        # RAG result
        "rag_evidence": rag_evidence,

        # Agentic RAG details
        "agentic_rag": {
            "retrieval_mode": rag_evidence.get(
                "retrieval_mode",
                "unknown",
            ),
            "agentic_rag": rag_evidence.get(
                "agentic_rag",
                False,
            ),
            "agentic_rag_error": rag_evidence.get(
                "agentic_rag_error",
            ),
            "trace": rag_evidence.get("agent_trace"),
            "provenance": rag_evidence.get("provenance"),
            "rounds_executed": rag_evidence.get("rounds_executed", 1),
            "sufficiency": rag_evidence.get("sufficiency", "unknown"),
        },

        # Final report
        "report": report,
    }


# ============================================================
# EXPLANATION
# ============================================================

def _explain(
    classification: dict,
    features: dict,
    evolution: dict,
) -> str:
    """Create a concise non-diagnostic explanation from the measured features."""

    parts = []

    prob = float(
        classification.get(
            "melanoma_probability",
            0.0,
        ) or 0.0
    )

    risk = classification.get(
        "risk_level",
        "low",
    )

    parts.append(
        f"Model-estimated melanoma risk is "
        f"{prob:.0%} ({risk} risk)."
    )

    diameter = features.get(
        "diameter",
        {},
    )

    diameter_mm = diameter.get(
        "max_diameter_mm"
    )

    if diameter.get(
        "available",
        False,
    ):

        if diameter.get(
            "exceeds_6mm"
        ):

            parts.append(
                f"Measured maximum diameter is "
                f"{diameter_mm:.1f} mm, "
                "above the conventional 6 mm "
                "ABCDE reference threshold."
            )

        else:

            parts.append(
                f"Measured maximum diameter is "
                f"{diameter_mm:.1f} mm."
            )

    else:

        parts.append(
            "Diameter is reported as a "
            "pixel-only estimate because no "
            "valid physical calibration was "
            "available, so it is not treated "
            "as a clinically validated mm "
            "measurement."
        )

    if features.get(
        "asymmetry",
        0.0,
    ) > 0.50:

        parts.append(
            "The normalized asymmetry score "
            "is relatively high."
        )

    if features.get(
        "border_irregularity",
        0.0,
    ) > 0.50:

        parts.append(
            "The normalized border-irregularity "
            "score is relatively high."
        )

    if features.get(
        "color",
        {},
    ).get(
        "index",
        0.0,
    ) > 0.50:

        parts.append(
            "The normalized color-variation "
            "index is relatively high."
        )

    if evolution.get(
        "available"
    ):

        if evolution.get(
            "change_detected"
        ):

            parts.append(
                "Changes compared with the "
                "previous analysis were detected."
            )

        else:

            parts.append(
                "No significant change was "
                "detected compared with the "
                "previous analysis."
            )

    else:

        parts.append(
            "A follow-up image of the same "
            "lesion can be used for evolution "
            "tracking."
        )

    return " ".join(parts)


# ============================================================
# SERIALIZE REPORT
# ============================================================

def serialize_report(
    result: dict,
) -> str:
    """Serialize the generated report as JSON."""

    return json.dumps(
        result["report"],
        default=str,
    )