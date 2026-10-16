"""
CEFM Frontend Helpers & Contract Validation

Pure-function utilities for Streamlit response handling, API error parsing,
persistence status classification, and unpersisted report export formatting.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def classify_persistence_status(result: dict[str, Any] | None) -> tuple[str, str | None]:
    """
    Classify the backend persistence status of an analysis result payload.

    Returns:
        tuple[str, str | None]:
            - Status category: 'PERSISTED', 'PARTIAL_SUCCESS', 'AMBIGUOUS_UNPERSISTED', or 'INVALID_RESPONSE'
            - Storage warning message if not fully persisted, or None if persisted.
    """
    if not isinstance(result, dict):
        return (
            "INVALID_RESPONSE",
            "Invalid or missing response payload from backend server."
        )

    status = result.get("status")
    persisted = result.get("persisted")

    # 1. Confirmed successful persistence:
    # Both status must be 'success' and persisted must explicitly be boolean True.
    if status == "success" and persisted is True:
        return ("PERSISTED", None)

    # 2. Confirmed partial success (inference succeeded, MongoDB persistence failed):
    # status must be 'partial_success' and persisted must explicitly be boolean False.
    if status == "partial_success" and persisted is False:
        warning = result.get(
            "storage_warning",
            "Analysis completed successfully, but results could not be persisted to the database. "
            "History and longitudinal tracking will not be available for this run."
        )
        return ("PARTIAL_SUCCESS", warning)

    # 3. Missing or contradictory persistence fields:
    # Fail safe: never assume saved.
    if persisted is None:
        warning = (
            "[WARNING] Persistence status unconfirmed: the backend response omitted the 'persisted' field. "
            "To prevent data corruption, this result is NOT saved to server database history."
        )
    elif status == "success" and persisted is not True:
        warning = (
            "[WARNING] Contradictory backend response: 'status' indicates success but 'persisted' is False or missing. "
            "This result is NOT saved to server database history."
        )
    elif status == "partial_success" and persisted is True:
        warning = (
            "[WARNING] Contradictory backend response: 'status' indicates partial success but 'persisted' is True. "
            "This result is NOT saved to server database history."
        )
    else:
        warning = (
            f"[WARNING] Unrecognized backend status ('{status}', persisted={persisted}). "
            "This result is NOT saved to server database history."
        )

    return ("AMBIGUOUS_UNPERSISTED", warning)


def should_save_to_history(persistence_status: str) -> bool:
    """Return True ONLY if persistence is verified and unambiguous."""
    return persistence_status == "PERSISTED"


def parse_api_error(
    status_code: int,
    response_json: dict[str, Any] | None = None,
    response_text: str = ""
) -> dict[str, Any]:
    """
    Parse an HTTP error from the FastAPI backend into safe, user-friendly UI details.

    Distinguishes:
        - HTTP 400: invalid or unsupported image / quality rejection
        - HTTP 413: upload exceeds file size limit
        - HTTP 503: database or backend service unavailable
        - HTTP 500 & other: internal pipeline error
    """
    resp_json = response_json if isinstance(response_json, dict) else {}
    detail = resp_json.get("detail", {})

    if status_code == 400:
        reason = None
        if isinstance(detail, dict):
            validation = detail.get("validation", {})
            reason = (
                detail.get("reason")
                or validation.get("calibration_rejection", {}).get("message")
                or validation.get("post_segmentation_quality", {}).get("reason")
                or validation.get("diameter_rejection", {}).get("message")
                or validation.get("reason")
                or detail.get("message")
            )
        elif isinstance(detail, str) and detail.strip():
            reason = detail.strip()

        if not reason:
            reason = "The uploaded file does not satisfy dermoscopic image quality or format requirements."

        return {
            "status_code": 400,
            "category": "INVALID_IMAGE",
            "title": "Image Rejected by Validation Gate",
            "message": f"Image rejected: {reason}",
            "rejection_reason": reason,
            "guidance": (
                "Please follow capture instructions: use even lighting, avoid glare and blur, "
                "and ensure either the 20 mm calibration marker or a clear millimeter ruler is visible."
            ),
            "is_image_rejection": True,
        }

    if status_code == 413:
        return {
            "status_code": 413,
            "category": "FILE_TOO_LARGE",
            "title": "Image File Exceeds Size Limit",
            "message": "The uploaded image exceeds the maximum permitted server size (25 MB).",
            "rejection_reason": "File size exceeds 25 MB limit.",
            "guidance": "Please compress or resize the dermoscopic photograph before uploading.",
            "is_image_rejection": False,
        }

    if status_code == 503:
        return {
            "status_code": 503,
            "category": "SERVICE_UNAVAILABLE",
            "title": "Clinical Database Service Unavailable",
            "message": "The backend database service is temporarily offline or unreachable.",
            "rejection_reason": None,
            "guidance": (
                "This is a backend infrastructure condition, not an image rejection. "
                "The server cannot currently query or store case histories. Please try again shortly."
            ),
            "is_image_rejection": False,
        }

    if status_code == 500:
        return {
            "status_code": 500,
            "category": "SERVER_ERROR",
            "title": "Analysis Pipeline Error",
            "message": "The analysis pipeline encountered an unexpected internal error processing this image.",
            "rejection_reason": None,
            "guidance": (
                "Please ensure the image is a valid dermoscopic photograph. "
                "If the error continues, please check server logs for diagnostic details."
            ),
            "is_image_rejection": False,
        }

    # Fallback for other status codes (e.g. 502, 504, 404, etc.)
    return {
        "status_code": status_code,
        "category": "SERVER_ERROR",
        "title": f"Backend Error (HTTP {status_code})",
        "message": f"Backend returned unexpected HTTP status code {status_code}.",
        "rejection_reason": None,
        "guidance": "Please verify backend connectivity and try again.",
        "is_image_rejection": False,
    }


def build_temp_case(
    patient: dict[str, Any],
    lesion: dict[str, Any],
    image_name: str,
    result: dict[str, Any]
) -> dict[str, Any]:
    """Construct a temporary case structure suitable for rendering and session storage."""
    symptoms = lesion.get("symptoms", [])
    symptoms_json = json.dumps(symptoms) if isinstance(symptoms, list) else str(symptoms or "[]")

    return {
        "patient_name": patient.get("full_name", "").strip(),
        "age": patient.get("age", ""),
        "gender": patient.get("gender", ""),
        "phone": patient.get("phone", ""),
        "visit_date": str(patient.get("visit_date", "")),
        "lesion_area": lesion.get("area", ""),
        "lesion_size": lesion.get("size", ""),
        "duration": lesion.get("duration", ""),
        "explanation": lesion.get("explanation", ""),
        "symptoms": symptoms_json,
        "image_name": image_name,
        "result_json": json.dumps(result) if isinstance(result, dict) else str(result),
        "persisted": False,
        "created_at": datetime.now().isoformat(),
    }


def generate_report_export(case: dict[str, Any]) -> str:
    """
    Format a clean clinical Markdown report from case data and available analysis results.
    Does not invent clinical findings or fabricate missing fields.
    """
    result_raw = case.get("result_json", "{}")
    result: dict[str, Any] = json.loads(result_raw) if isinstance(result_raw, str) else (result_raw or {})

    symptoms_raw = case.get("symptoms", "[]")
    symptoms: list[str] = json.loads(symptoms_raw) if isinstance(symptoms_raw, str) else (symptoms_raw or [])

    clf = result.get("classification", {})
    feat = result.get("features") or result.get("abcde", {})
    color = feat.get("color", {}) if isinstance(feat.get("color"), dict) else {}
    rep = result.get("report", {})
    expl = result.get("explanation") or rep.get("explanation", {})
    evolution = result.get("evolution", {})

    pred = clf.get("predicted_class", clf.get("prediction", "Unavailable"))
    conf = float(clf.get("melanoma_probability", clf.get("confidence", 0)) or 0)
    risk = rep.get("overall_risk", clf.get("risk_level", "Unavailable"))

    a_score = feat.get("asymmetry")
    b_score = feat.get("border_irregularity")
    c_score = feat.get("color_index", color.get("index"))
    d_mm = feat.get("diameter_mm")
    d_px = feat.get("diameter_pixels")

    if d_mm is not None:
        diameter_str = f"{float(d_mm):.2f} mm"
        if float(d_mm) > 6.0:
            diameter_str += " (WARNING: Exceeds 6.0 mm threshold)"
    elif d_px is not None:
        diameter_str = f"{float(d_px):.0f} px (Physical diameter unavailable - calibration reference not detected)"
    else:
        diameter_str = "Unavailable"

    if evolution.get("available"):
        change_text = "Significant change detected" if evolution.get("change_detected") else "No significant change detected"
        evolution_str = f"{change_text} - {evolution.get('description', '')}"
    else:
        evolution_str = evolution.get("reason", "No previous image available for longitudinal comparison.")

    full_report = rep.get("full_report_text", "Structured report unavailable.")
    recommendation = rep.get("recommendation", "Clinical evaluation recommended.")
    summary_expl = expl.get("summary", "Clinical feature explanation unavailable.")

    a_text = f"{float(a_score):.2f} (0.00 = symmetrical, 1.00 = highly asymmetrical)" if a_score is not None else "Unavailable"
    b_text = f"{float(b_score):.2f} (0.00 = smooth, 1.00 = highly irregular)" if b_score is not None else "Unavailable"
    c_text = f"{float(c_score):.2f} (0.00 = uniform, 1.00 = multi-chromatic)" if c_score is not None else "Unavailable"

    export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# CEFM Clinical Melanoma Assessment Report
**Export Generated:** {export_time}  
**Image File:** {case.get('image_name', '-')}  
**Persistence Status:** [WARNING] NOT SAVED TO SERVER DATABASE (Exported from temporary session memory)

---

## 1. Patient & Case Information
- **Patient Name:** {case.get('patient_name', '-')}
- **Age / Gender:** {case.get('age', '-')} yrs / {case.get('gender', '-')}
- **Contact Phone:** {case.get('phone') or 'Not provided'}
- **Visit Date:** {case.get('visit_date', '-')}
- **Lesion Body Area:** {case.get('lesion_area', '-')}
- **Patient-Reported Approximate Size:** {case.get('lesion_size') or 'Not reported'}
- **Duration Observed:** {case.get('duration') or 'Not reported'}
- **Reported Symptoms:** {', '.join(symptoms) if symptoms else 'None reported'}
- **Clinical Notes / Symptoms Description:**
  {case.get('explanation', '-')}

---

## 2. Artificial Intelligence Assessment
- **Predicted Class:** {pred}
- **Melanoma Probability (Model Confidence):** {conf * 100:.1f}%
- **Overall Clinical Risk Level:** {risk}

---

## 3. Quantitative ABCDE Dermoscopic Features
- **Asymmetry (A):** {a_text}
- **Border Irregularity (B):** {b_text}
- **Color Variation (C):** {c_text}
- **Diameter (D):** {diameter_str}
- **Evolution (E):** {evolution_str}

---

## 4. Structured Clinical Report
{full_report}

**Clinical Recommendation:**  
{recommendation}

---

## 5. Clinical Feature Reasoning
{summary_expl}

---

## 6. Important Clinical & Technical Disclaimers
1. **Storage Status Notice**: This assessment was NOT saved to the permanent server database because database persistence failed or was unconfirmed. It was exported directly from temporary browser session memory. This record cannot be retrieved from server history at a later date.
2. **Clinical Decision Support Notice**: This report was generated by an artificial intelligence decision-support research system (CEFM). It does NOT constitute a definitive medical diagnosis. All findings, measurements, and risk estimates must be independently validated by a qualified dermatologist or medical practitioner.
"""
