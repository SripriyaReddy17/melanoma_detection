import io
import json
import shutil
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response

from app.core.config import (
    DATA_DIR,
    MAX_UPLOAD_SIZE_MB,
    MAX_UPLOAD_BYTES,
    ALLOWED_EXTENSIONS,
    sanitize_log_message,
)
from app.database import (
    cases_collection,
    analyses_collection,
)
from app.services.analysis import analyze_image
from app.services.visualization import render_view


# ============================================================
# UPLOAD STORAGE
# ============================================================

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter()


ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


# ============================================================
# IMAGE MAGIC-BYTE SIGNATURES
# ============================================================
# Used to verify that the uploaded file is a real image,
# regardless of the Content-Type header sent by the client.
# ============================================================

_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",     # JPEG / JFIF
    b"\x89PNG":      "png",      # PNG
}


# ============================================================
# MONGODB INDEXES
# ============================================================

def init_mongodb_indexes():
    """Ensure indexes exist for cases and analyses when MongoDB is available."""
    try:
        cases_collection.create_index(
            [("label", 1), ("created_at", -1)]
        )
        analyses_collection.create_index(
            [("case_id", 1), ("created_at", -1)]
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB WARNING] Could not create indexes: {safe_msg}")


# ============================================================
# CASE HELPERS
# ============================================================

def _resolve_patient_case(
    case_id: str | None = None,
    patient_name: str | None = None,
):
    """
    Return an existing MongoDB case or create a new one.

    The case label acts as the patient/case identifier so that
    multiple uploaded images can be grouped together for
    Evolution (E) tracking.
    """

    # --------------------------------------------------------
    # Existing case requested
    # --------------------------------------------------------

    if case_id:
        try:
            case = cases_collection.find_one(
                {"_id": case_id}
            )
        except Exception as exc:
            safe_msg = sanitize_log_message(str(exc))
            print(f"[MongoDB WARNING] Case lookup failed: {safe_msg}")
            raise HTTPException(
                status_code=503,
                detail="Database service is temporarily unavailable.",
            )

        if not case:
            raise HTTPException(
                status_code=404,
                detail=f"Case {case_id} not found.",
            )

        return case

    # --------------------------------------------------------
    # Resolve case label
    # --------------------------------------------------------

    patient_label = (
        patient_name or ""
    ).strip()

    if not patient_label:
        patient_label = (
            f"Case {datetime.utcnow():%Y-%m-%d %H:%M}"
        )

    # --------------------------------------------------------
    # Find most recent case with same label
    # --------------------------------------------------------

    try:
        case = cases_collection.find_one(
            {"label": patient_label},
            sort=[
                ("created_at", -1)
            ],
        )
        if case is not None:
            return case
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB WARNING] Case search by label failed: {safe_msg}")
        return None

    # --------------------------------------------------------
    # Create new case
    # --------------------------------------------------------

    case_id = uuid.uuid4().hex

    case = {
        "_id": case_id,
        "label": patient_label,
        "created_at": datetime.utcnow(),
    }

    try:
        cases_collection.insert_one(case)
        return case
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB WARNING] Case creation failed: {safe_msg}")
        return None


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health_check():

    from app.database import check_mongodb

    mongo_ok = check_mongodb()

    return {
        "status": "ok" if mongo_ok else "degraded",
        "message": "CEFM Backend is running",
        "mongodb": "connected" if mongo_ok else "unreachable",
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    case_id: str | None = Form(None),
    case_label: str | None = Form(None),
    patient_name: str | None = Form(None),
    remove_hair: bool = Form(False),
    pixels_per_mm: float | None = Form(None),
):
    """
    Upload an image and run the full melanoma pipeline.

    Pipeline:

    validation
        -> image type
        -> calibration
        -> segmentation
        -> ABCDE
        -> classification
        -> evolution
        -> RAG
        -> report

    Image metadata and analysis results are stored in MongoDB.
    """


    # ========================================================
    # 1. VALIDATE FILE TYPE (content-type + extension)
    # ========================================================

    content_type = (image.content_type or "").lower().strip()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG images "
                "are supported."
            ),
        )

    extension = (
        Path(
            image.filename or "image.jpg"
        ).suffix.lower()
        or ".jpg"
    )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File extension '{extension}' is not "
                f"allowed. Accepted: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )


    # ========================================================
    # 1b. ENFORCE UPLOAD SIZE LIMIT (FAST-PATH + BOUNDED STREAM)
    # ========================================================

    # Fast-check declared content-length header if provided
    cl_header = image.headers.get("content-length")
    if cl_header and cl_header.isdigit() and int(cl_header) > MAX_UPLOAD_BYTES:
        await image.close()
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB.",
        )

    chunks = []
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB bounded reads

    try:
        while True:
            chunk = await image.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                await image.close()
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB.",
                )
            chunks.append(chunk)
    finally:
        await image.close()

    if total_bytes == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    file_bytes = b"".join(chunks)


    # ========================================================
    # 1c. MAGIC-BYTE VERIFICATION & IMAGE INTEGRITY
    # ========================================================
    # Ensure the file content matches a known image signature
    # regardless of what the client claims in Content-Type.
    # ========================================================

    header = file_bytes[:8]
    is_real_image = any(
        header.startswith(sig)
        for sig in _MAGIC_BYTES
    )

    if not is_real_image:
        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file does not appear to be a "
                "valid JPEG or PNG image."
            ),
        )

    # In-memory structural validation to reject malformed image data early
    try:
        with PILImage.open(io.BytesIO(file_bytes)) as pil_img:
            pil_img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file contains invalid or corrupted image data.",
        )


    # ========================================================
    # 2. SAVE IMAGE
    # ========================================================

    file_id = uuid.uuid4().hex

    stored_path = (
        UPLOAD_DIR /
        f"{file_id}{extension}"
    )

    try:
        stored_path.write_bytes(file_bytes)

    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[Upload ERROR] Could not save upload to disk: {safe_msg}")
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded image for processing.",
        )


    # ========================================================
    # 3. RESOLVE CASE
    # ========================================================

    resolved_patient_name = (
        patient_name
        or case_label
        or ""
    ).strip()

    resolved_case_id = None

    try:
        case = _resolve_patient_case(
            case_id=case_id,
            patient_name=resolved_patient_name,
        )
        resolved_case_id = case["_id"] if case else None

    except HTTPException:
        stored_path.unlink(
            missing_ok=True
        )
        raise

    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB WARNING] Case resolution encountered an error: {safe_msg}")
        resolved_case_id = None


    # ========================================================
    # 4. GET PREVIOUS ANALYSIS FOR EVOLUTION
    # ========================================================

    previous = None

    if resolved_case_id:
        try:
            previous = analyses_collection.find_one(
                {
                    "case_id": resolved_case_id,
                    "is_valid": True,
                },
                sort=[
                    ("created_at", -1)
                ],
            )
        except Exception as exc:
            safe_msg = sanitize_log_message(str(exc))
            print(f"[MongoDB WARNING] Previous analysis lookup failed: {safe_msg}")
            previous = None


    previous_features = None


    if previous:

        previous_features = {
            "asymmetry": previous.get(
                "asymmetry"
            ),

            "border_irregularity": previous.get(
                "border_irregularity"
            ),

            "color_h": previous.get(
                "color_h"
            ),

            "color_s": previous.get(
                "color_s"
            ),

            "color_v": previous.get(
                "color_v"
            ),

            "color_index": previous.get(
                "color_index"
            ),

            "diameter_pixels": previous.get(
                "diameter_pixels"
            ),

            "diameter_mm": previous.get(
                "diameter_mm"
            ),
        }


    # ========================================================
    # 5. RUN COMPLETE MELANOMA ANALYSIS
    # ========================================================

    try:

        result = analyze_image(
            str(stored_path),
            image.filename or "",
            previous_features,
            remove_hair_flag=remove_hair,
            pixels_per_mm=pixels_per_mm,
        )

    except Exception as exc:

        # Print the complete traceback in the terminal for debugging.
        traceback.print_exc()

        stored_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail="Analysis pipeline encountered an error. Please try again.",
        )


    # ========================================================
    # 6. HANDLE INVALID IMAGE
    # ========================================================

    if result["status"] == "invalid":

        stored_path.unlink(
            missing_ok=True
        )

        validation = result.get(
            "validation",
            {},
        )

        if not isinstance(
            validation,
            dict,
        ):
            validation = {}


        reason = None


        # Calibration rejection
        calibration_rejection = validation.get(
            "calibration_rejection",
            {},
        )

        if isinstance(
            calibration_rejection,
            dict,
        ):

            reason = (
                calibration_rejection.get(
                    "message"
                )
                or reason
            )


        # Post segmentation quality
        if not reason:

            post_quality = validation.get(
                "post_segmentation_quality",
                {},
            )

            if isinstance(
                post_quality,
                dict,
            ):

                reason = (
                    post_quality.get(
                        "reason"
                    )
                    or reason
                )


        # Diameter rejection
        if not reason:

            diameter_rejection = validation.get(
                "diameter_rejection",
                {},
            )

            if isinstance(
                diameter_rejection,
                dict,
            ):

                reason = (
                    diameter_rejection.get(
                        "message"
                    )
                    or reason
                )


        # Basic validation
        if not reason:

            basic_reason = validation.get(
                "reason"
            )

            if (
                basic_reason
                and basic_reason
                != "All basic image quality checks passed."
            ):

                reason = basic_reason


        if not reason:

            reason = (
                "The uploaded image did not "
                "satisfy the required quality "
                "or calibration checks."
            )


        if isinstance(
            reason,
            list,
        ):

            reason = " ".join(
                str(item)
                for item in reason
            )


        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Image validation failed."
                ),
                "reason": str(reason),
                "validation": validation,
                "rejection_stage": result.get(
                    "rejection_stage"
                ),
                "segmentation": result.get(
                    "segmentation",
                    {},
                ),
                "segmentation_url": (
                    f"/api/v1/segmentation/{file_id}"
                    if result.get("segmentation_path")
                    else None
                ),
            },
        )


    # ========================================================
    # 7. EXTRACT RESULTS
    # ========================================================

    features = result["features"]

    classification = result[
        "classification"
    ]

    evolution = result[
        "evolution"
    ]

    response_evolution = (
        evolution
        if previous is not None
        else None
    )


    # ========================================================
    # 8. CREATE MONGODB ANALYSIS DOCUMENT
    # ========================================================

    analysis_id = uuid.uuid4().hex

    created_at = datetime.utcnow()


    record = {

        "_id": analysis_id,

        "case_id": resolved_case_id,

        "original_filename": (
            image.filename
        ),

        "stored_path": str(
            stored_path
        ),

        "segmentation_path": result.get(
            "segmentation_path"
        ),

        "segmentation": result.get(
            "segmentation",
            {},
        ),

        "is_valid": True,


        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        "image_type": result.get(
            "image_type",
            {},
        ),


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        "predicted_class": classification.get(
            "predicted_class"
        ),

        "melanoma_probability": classification.get(
            "melanoma_probability"
        ),

        "risk_level": classification.get(
            "risk_level"
        ),


        # ----------------------------------------------------
        # ABCDE
        # ----------------------------------------------------

        "asymmetry": features.get(
            "asymmetry"
        ),

        "border_irregularity": features.get(
            "border_irregularity"
        ),

        "color_h": features.get(
            "color_h"
        ),

        "color_s": features.get(
            "color_s"
        ),

        "color_v": features.get(
            "color_v"
        ),

        "color_index": features.get(
            "color_index"
        ),

        "diameter_pixels": features.get(
            "diameter_pixels"
        ),

        "diameter_mm": features.get(
            "diameter_mm"
        ),


        # ----------------------------------------------------
        # Evolution
        # ----------------------------------------------------

        "evolution_available": evolution.get(
            "available",
            False,
        ),

        "evolution_change_detected": evolution.get(
            "change_detected",
            False,
        ),

        "evolution": evolution,

        # ----------------------------------------------------
        # Multimodal / Contrastive Alignment
        # ----------------------------------------------------

        "multimodal": result.get(
            "multimodal",
            {},
        ),

        "contrastive_alignment": result.get(
            "contrastive_alignment",
            {},
        ),

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        "rag_evidence": result.get(
            "rag_evidence",
            {},
        ),


        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        "explanation": result.get(
            "explanation",
            {},
        ),


        # ----------------------------------------------------
        # Complete report
        # ----------------------------------------------------

        "report": result.get(
            "report",
            {},
        ),


        "created_at": created_at,
    }


    # ========================================================
    # 9. SAVE ANALYSIS TO MONGODB (SAFE PERSISTENCE)
    # ========================================================

    persisted = False
    db_analysis_id = None

    if resolved_case_id:
        try:
            analyses_collection.insert_one(record)
            persisted = True
            db_analysis_id = analysis_id
        except Exception as exc:
            safe_msg = sanitize_log_message(str(exc))
            print(
                f"[MongoDB WARNING] Could not persist "
                f"analysis {analysis_id}: {safe_msg}"
            )
            persisted = False
            # Clean up stored file since no MongoDB record references it
            stored_path.unlink(missing_ok=True)
    else:
        # Case was not persisted, so analysis cannot be linked in MongoDB
        stored_path.unlink(missing_ok=True)


    # ========================================================
    # 10. RESPONSE
    # ========================================================

    response_payload = {
        "status": "success" if persisted else "partial_success",
        "persisted": persisted,
        "storage_status": "saved" if persisted else "persistence_failed",
        "analysis_id": db_analysis_id,
        "case_id": resolved_case_id if persisted else None,
        "image_url": (
            f"/api/v1/images/{db_analysis_id}"
            if persisted
            else None
        ),
        "segmentation_url": (
            f"/api/v1/images/{db_analysis_id}/segmentation"
            if persisted and result.get("segmentation_path")
            else None
        ),
        "segmentation": result.get(
            "segmentation",
            {},
        ),
        "filename": image.filename,
        "validation": result[
            "validation"
        ],
        "image_type": result[
            "image_type"
        ],
        "calibration": result.get(
            "calibration"
        ),
        "abcde": {
            "asymmetry": features.get("asymmetry"),
            "border_irregularity": features.get("border_irregularity"),
            "color_h": features.get("color_h"),
            "color_s": features.get("color_s"),
            "color_v": features.get("color_v"),
            "color_index": features.get("color_index"),
            "diameter_pixels": features.get("diameter_pixels"),
            "diameter_mm": features.get("diameter_mm"),
        },
        "classification": classification,
        "evolution": response_evolution,
        "multimodal": result.get(
            "multimodal",
            {},
        ),
        "contrastive_alignment": result.get(
            "contrastive_alignment",
            {},
        ),
        "explanation": result[
            "explanation"
        ],
        "rag_evidence": result.get(
            "rag_evidence",
            {},
        ),
        "report": result[
            "report"
        ],
    }

    if not persisted:
        response_payload["storage_warning"] = (
            "Analysis completed successfully, but results could not be persisted to the database. "
            "History and longitudinal tracking will not be available for this run."
        )

    return response_payload


# ============================================================
# CREATE CASE
# ============================================================

@router.post("/cases")
def create_case(
    label: str | None = Form(None),
):

    case_id = uuid.uuid4().hex

    case_label = (
        label.strip()
        if label
        else None
    )

    if not case_label:
        case_label = (
            f"Case {datetime.utcnow():%Y-%m-%d %H:%M}"
        )

    case = {
        "_id": case_id,
        "label": case_label,
        "created_at": datetime.utcnow(),
    }

    try:
        cases_collection.insert_one(
            case
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to create case: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    return {
        "case_id": case_id,
        "label": case_label,
    }


# ============================================================
# LIST CASES
# ============================================================

@router.get("/cases")
def list_cases():

    try:
        cases = list(
            cases_collection.find(
                {}
            ).sort(
                "created_at",
                -1,
            )
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to list cases: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    response = []

    for case in cases:
        try:
            number_of_analyses = (
                analyses_collection.count_documents(
                    {
                        "case_id": case["_id"]
                    }
                )
            )
        except Exception:
            number_of_analyses = 0

        created_at = case.get(
            "created_at"
        )

        response.append(
            {
                "case_id": case["_id"],
                "label": case.get(
                    "label"
                ),
                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),
                "num_analyses": (
                    number_of_analyses
                ),
            }
        )

    return response


# ============================================================
# GET SINGLE CASE
# ============================================================

@router.get("/cases/{case_id}")
def get_case(
    case_id: str,
):

    try:
        case = cases_collection.find_one(
            {
                "_id": case_id
            }
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get case: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    if not case:
        raise HTTPException(
            status_code=404,
            detail="Case not found.",
        )

    try:
        analyses = list(
            analyses_collection.find(
                {
                    "case_id": case_id
                }
            ).sort(
                "created_at",
                1,
            )
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get case analyses: {safe_msg}")
        analyses = []

    analysis_response = []

    for analysis in analyses:

        created_at = analysis.get(
            "created_at"
        )

        analysis_response.append(
            {
                "analysis_id": analysis["_id"],
                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),
                "predicted_class": analysis.get(
                    "predicted_class"
                ),
                "melanoma_probability": analysis.get(
                    "melanoma_probability"
                ),
                "risk_level": analysis.get(
                    "risk_level"
                ),
                "diameter_pixels": analysis.get(
                    "diameter_pixels"
                ),
                "diameter_mm": analysis.get(
                    "diameter_mm"
                ),
                "asymmetry": analysis.get(
                    "asymmetry"
                ),
                "border_irregularity": analysis.get(
                    "border_irregularity"
                ),
                "evolution_change_detected": analysis.get(
                    "evolution_change_detected",
                    False,
                ),
                "image_url": (
                    f"/api/v1/images/"
                    f"{analysis['_id']}"
                ),
            }
        )

    created_at = case.get(
        "created_at"
    )

    return {
        "case_id": case["_id"],
        "label": case.get(
            "label"
        ),
        "created_at": (
            created_at.isoformat()
            if created_at
            else None
        ),
        "analyses": analysis_response,
    }


# ============================================================
# LIST ANALYSES
# ============================================================

@router.get("/analyses")
def list_analyses(
    limit: int = 50,
):

    # Prevent unreasonable requests
    limit = max(
        1,
        min(limit, 500),
    )

    try:
        rows = list(
            analyses_collection.find(
                {}
            ).sort(
                "created_at",
                -1,
            ).limit(
                limit
            )
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to list analyses: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    response = []

    for analysis in rows:

        created_at = analysis.get(
            "created_at"
        )

        response.append(
            {
                "analysis_id": analysis["_id"],
                "case_id": analysis.get(
                    "case_id"
                ),
                "filename": analysis.get(
                    "original_filename"
                ),
                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),
                "predicted_class": analysis.get(
                    "predicted_class"
                ),
                "melanoma_probability": analysis.get(
                    "melanoma_probability"
                ),
                "risk_level": analysis.get(
                    "risk_level"
                ),
                "diameter_pixels": analysis.get(
                    "diameter_pixels"
                ),
                "diameter_mm": analysis.get(
                    "diameter_mm"
                ),
                "image_url": (
                    f"/api/v1/images/"
                    f"{analysis['_id']}"
                ),
            }
        )

    return response


# ============================================================
# GET SINGLE ANALYSIS
# ============================================================

@router.get("/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
):

    try:
        analysis = analyses_collection.find_one(
            {
                "_id": analysis_id
            }
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get analysis: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    created_at = analysis.get(
        "created_at"
    )

    return {
        "analysis_id": analysis["_id"],
        "case_id": analysis.get(
            "case_id"
        ),
        "filename": analysis.get(
            "original_filename"
        ),
        "created_at": (
            created_at.isoformat()
            if created_at
            else None
        ),
        "image_url": (
            f"/api/v1/images/"
            f"{analysis['_id']}"
        ),
        "abcde": {
            "asymmetry": analysis.get(
                "asymmetry"
            ),
            "border_irregularity": analysis.get(
                "border_irregularity"
            ),
            "color_h": analysis.get(
                "color_h"
            ),
            "color_s": analysis.get(
                "color_s"
            ),
            "color_v": analysis.get(
                "color_v"
            ),
            "color_index": analysis.get(
                "color_index"
            ),
            "diameter_pixels": analysis.get(
                "diameter_pixels"
            ),
            "diameter_mm": analysis.get(
                "diameter_mm"
            ),
        },
        "classification": {
            "predicted_class": analysis.get(
                "predicted_class"
            ),
            "melanoma_probability": analysis.get(
                "melanoma_probability"
            ),
            "risk_level": analysis.get(
                "risk_level"
            ),
        },
        "evolution": analysis.get(
            "evolution"
        ),
        "multimodal": analysis.get(
            "multimodal",
            {},
        ),
        "contrastive_alignment": analysis.get(
            "contrastive_alignment",
            {},
        ),
        "rag_evidence": analysis.get(
            "rag_evidence",
            {},
        ),
        "explanation": analysis.get(
            "explanation",
            {},
        ),
        "report": analysis.get(
            "report"
        ),
    }


# ============================================================
# GET STORED IMAGE
# ============================================================

@router.get("/images/{analysis_id}")
def get_image(
    analysis_id: str,
):

    try:
        analysis = analyses_collection.find_one(
            {
                "_id": analysis_id
            }
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get image for analysis: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    stored_path = analysis.get(
        "stored_path"
    )

    if (
        not stored_path
        or not Path(stored_path).exists()
    ):
        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    return FileResponse(
        stored_path
    )


# ============================================================
# SEGMENTATION OVERLAY IMAGE
# ============================================================

@router.get("/images/{analysis_id}/segmentation")
def get_segmentation_image(
    analysis_id: str,
):
    """
    Serve the segmentation overlay PNG for an analysis.
    The overlay shows the lesion mask (red fill + cyan contour)
    on the original image. Available for ALL analyses, including
    those rejected by calibration or quality gates.
    """

    try:
        analysis = analyses_collection.find_one(
            {"_id": analysis_id}
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get segmentation image record: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    seg_path = analysis.get("segmentation_path")

    if not seg_path or not Path(seg_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Segmentation image not available.",
        )

    return FileResponse(
        seg_path,
        media_type="image/png",
        filename=f"segmentation_{analysis_id}.png",
    )


# ============================================================
# XAI VISUAL EVIDENCE VIEWS
# ============================================================

@router.get(
    "/analyses/{analysis_id}/view/{view_name}"
)
def get_analysis_view(
    analysis_id: str,
    view_name: str,
    scale_px_per_mm: float | None = None,
):

    try:
        analysis = analyses_collection.find_one(
            {
                "_id": analysis_id
            }
        )
    except Exception as exc:
        safe_msg = sanitize_log_message(str(exc))
        print(f"[MongoDB ERROR] Failed to get view record: {safe_msg}")
        raise HTTPException(
            status_code=503,
            detail="Database service is temporarily unavailable.",
        )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )


    stored_path = analysis.get(
        "stored_path"
    )


    if (
        not stored_path
        or not Path(stored_path).exists()
    ):

        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )


    try:

        png = render_view(
            view_name,
            stored_path,
            scale_px_per_mm,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate "
                f"visualization: {exc}"
            ),
        )


    return Response(
        content=png,
        media_type="image/png",
    )