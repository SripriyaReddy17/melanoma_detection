"""
Tests for Streamlit frontend response contract, persistence handling,
API error mapping, and report export utilities.

Exercises real frontend functions in frontend_helpers.py and verifies
contract compliance with FastAPI backend without requiring a live browser.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from frontend_helpers import (
    classify_persistence_status,
    should_save_to_history,
    parse_api_error,
    build_temp_case,
    generate_report_export,
)


# ============================================================
# HELPER: TEMPORARY SQLITE DATABASE
# ============================================================

def create_test_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_user_id INTEGER NOT NULL,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            visit_date TEXT,
            lesion_area TEXT,
            lesion_size TEXT,
            duration TEXT,
            symptoms TEXT,
            explanation TEXT,
            image_name TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def insert_case_record(conn: sqlite3.Connection, user_id: int, case: dict[str, Any]):
    conn.execute("""
        INSERT INTO cases (
            patient_user_id, patient_name, age, gender, phone, visit_date,
            lesion_area, lesion_size, duration, symptoms, explanation,
            image_name, result_json, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        case["patient_name"], case["age"], case["gender"], case.get("phone", ""),
        case["visit_date"], case["lesion_area"], case.get("lesion_size", ""),
        case.get("duration", ""), case.get("symptoms", "[]"), case["explanation"],
        case["image_name"], case["result_json"], case.get("created_at", "2026-10-16T12:00:00")
    ))
    conn.commit()


# ============================================================
# TASK 1: PERSISTENCE CONTRACT CLASSIFICATION TESTS
# ============================================================

def test_confirmed_persistence_handling(tmp_path):
    """
    Verify that when status='success' and persisted=True,
    the response is classified as PERSISTED, should_save_to_history is True,
    and a record is safely stored in SQLite history.
    """
    backend_response = {
        "status": "success",
        "persisted": True,
        "storage_status": "saved",
        "analysis_id": "mongo_id_12345",
        "case_id": "case_123",
        "image_url": "/api/v1/images/mongo_id_12345",
        "segmentation_url": "/api/v1/images/mongo_id_12345/segmentation",
        "filename": "derm.jpg",
        "classification": {"predicted_class": "non_mel", "melanoma_probability": 0.12},
        "abcde": {"asymmetry": 0.2, "diameter_mm": 4.5},
        "report": {"full_report_text": "Benign nevus."},
    }

    status_category, warning = classify_persistence_status(backend_response)
    assert status_category == "PERSISTED"
    assert warning is None
    assert should_save_to_history(status_category) is True

    # Test database write
    db_path = tmp_path / "test_persisted.db"
    conn = create_test_db(db_path)
    patient = {"full_name": "Jane Doe", "age": 30, "gender": "Female", "phone": "555-1234", "visit_date": "2026-10-16"}
    lesion = {"area": "Back", "size": "4 mm", "duration": "2 months", "symptoms": ["Itching"], "explanation": "Checkup"}

    case = build_temp_case(patient, lesion, "derm.jpg", backend_response)
    insert_case_record(conn, 1, case)

    row = conn.execute("SELECT * FROM cases WHERE patient_user_id=1").fetchone()
    assert row is not None
    conn.close()


def test_partial_success_handling(tmp_path):
    """
    Verify that when status='partial_success' and persisted=False,
    the response is classified as PARTIAL_SUCCESS, storage warning is returned,
    should_save_to_history is False, and NO record is written to SQLite history.
    """
    backend_response = {
        "status": "partial_success",
        "persisted": False,
        "storage_status": "persistence_failed",
        "storage_warning": "Analysis completed successfully, but results could not be persisted to the database.",
        "analysis_id": None,
        "case_id": None,
        "image_url": None,
        "segmentation_url": None,
        "filename": "derm.jpg",
        "classification": {"predicted_class": "non_mel", "melanoma_probability": 0.12},
        "abcde": {"asymmetry": 0.2, "diameter_mm": 4.5},
        "report": {"full_report_text": "Benign nevus."},
    }

    status_category, warning = classify_persistence_status(backend_response)
    assert status_category == "PARTIAL_SUCCESS"
    assert warning is not None
    assert "could not be persisted" in warning
    assert should_save_to_history(status_category) is False

    # Verify that database write is prevented
    db_path = tmp_path / "test_partial.db"
    conn = create_test_db(db_path)
    if should_save_to_history(status_category):
        insert_case_record(conn, 1, {})

    count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    assert count == 0
    conn.close()


def test_missing_persistence_field_fails_safely(tmp_path):
    """
    Verify that if the 'persisted' field is omitted or None,
    the response is classified as AMBIGUOUS_UNPERSISTED and NOT saved to history.
    """
    # Payload omitting 'persisted' entirely
    missing_persisted_payload = {
        "status": "success",
        "analysis_id": "legacy_or_buggy_id",
        "classification": {"predicted_class": "non_mel"},
    }

    status_category, warning = classify_persistence_status(missing_persisted_payload)
    assert status_category == "AMBIGUOUS_UNPERSISTED"
    assert warning is not None
    assert "omitted the 'persisted' field" in warning
    assert should_save_to_history(status_category) is False

    # Payload with persisted=None
    null_persisted_payload = {
        "status": "success",
        "persisted": None,
        "classification": {"predicted_class": "non_mel"},
    }

    status_category_null, warning_null = classify_persistence_status(null_persisted_payload)
    assert status_category_null == "AMBIGUOUS_UNPERSISTED"
    assert should_save_to_history(status_category_null) is False

    # Ensure no database writes occur
    db_path = tmp_path / "test_missing.db"
    conn = create_test_db(db_path)
    if should_save_to_history(status_category):
        insert_case_record(conn, 1, {})
    count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    assert count == 0
    conn.close()


def test_contradictory_persistence_fields_fail_safely():
    """
    Verify that contradictory or unrecognized status/persistence combinations
    fail safely and do not trigger history persistence.
    """
    # 1. status='success' but persisted=False
    c1, w1 = classify_persistence_status({"status": "success", "persisted": False})
    assert c1 == "AMBIGUOUS_UNPERSISTED"
    assert should_save_to_history(c1) is False

    # 2. status='partial_success' but persisted=True
    c2, w2 = classify_persistence_status({"status": "partial_success", "persisted": True})
    assert c2 == "AMBIGUOUS_UNPERSISTED"
    assert should_save_to_history(c2) is False

    # 3. Unrecognized status with persisted=True
    c3, w3 = classify_persistence_status({"status": "unknown_state", "persisted": True})
    assert c3 == "AMBIGUOUS_UNPERSISTED"
    assert should_save_to_history(c3) is False

    # 4. Non-dict or None input
    c4, w4 = classify_persistence_status(None)
    assert c4 == "INVALID_RESPONSE"
    assert should_save_to_history(c4) is False


# ============================================================
# TASK 3: API STATUS CODE MAPPING TESTS
# ============================================================

def test_api_status_code_mapping_invalid_image():
    """Verify HTTP 400 maps to INVALID_IMAGE and extracts rejection reasons."""
    resp_payload = {
        "detail": {
            "message": "Image validation failed.",
            "reason": "Calibration marker not found or blurred.",
            "validation": {"calibration_rejection": {"message": "Calibration marker not found."}}
        }
    }
    err = parse_api_error(400, resp_payload)
    assert err["status_code"] == 400
    assert err["category"] == "INVALID_IMAGE"
    assert err["is_image_rejection"] is True
    assert "Calibration marker not found" in err["message"]
    assert "even lighting" in err["guidance"]


def test_api_status_code_mapping_file_too_large():
    """Verify HTTP 413 maps to FILE_TOO_LARGE and mentions 25 MB limit."""
    resp_payload = {"detail": "File exceeds maximum allowed size of 25 MB."}
    err = parse_api_error(413, resp_payload)
    assert err["status_code"] == 413
    assert err["category"] == "FILE_TOO_LARGE"
    assert err["is_image_rejection"] is False
    assert "25 MB" in err["message"]


def test_api_status_code_mapping_service_unavailable():
    """
    Verify HTTP 503 maps to SERVICE_UNAVAILABLE.
    CRITICAL: Confirms database failure is NEVER presented as an image rejection.
    """
    resp_payload = {"detail": "Database service is temporarily unavailable."}
    err = parse_api_error(503, resp_payload)
    assert err["status_code"] == 503
    assert err["category"] == "SERVICE_UNAVAILABLE"
    assert err["is_image_rejection"] is False
    assert "database" in err["title"].lower()
    assert "not an image rejection" in err["guidance"].lower()


def test_api_status_code_mapping_server_error():
    """Verify HTTP 500 maps to safe SERVER_ERROR without exposing internal traces."""
    err = parse_api_error(500, {"detail": "Internal pipeline exception at /var/www/internal_script.py"})
    assert err["status_code"] == 500
    assert err["category"] == "SERVER_ERROR"
    assert err["is_image_rejection"] is False
    assert "unexpected internal error" in err["message"].lower()
    # Confirm raw internal path was NOT propagated into user-facing message
    assert "/var/www" not in err["message"]


# ============================================================
# TASK 2: UNSAVED RESULTS RETENTION AND EXPORT TESTS
# ============================================================

def test_build_temp_case_structure():
    """Verify build_temp_case constructs valid case dictionary."""
    patient = {"full_name": "Alice Smith", "age": 42, "gender": "Female", "phone": "123-456", "visit_date": "2026-10-16"}
    lesion = {"area": "Forearm", "size": "5 mm", "duration": "1 year", "symptoms": ["Bleeding", "Pain"], "explanation": "Growing lesion"}
    result = {"classification": {"predicted_class": "mel", "melanoma_probability": 0.85}}

    temp_case = build_temp_case(patient, lesion, "lesion_01.jpg", result)

    assert temp_case["patient_name"] == "Alice Smith"
    assert temp_case["age"] == 42
    assert temp_case["gender"] == "Female"
    assert temp_case["lesion_area"] == "Forearm"
    assert temp_case["image_name"] == "lesion_01.jpg"
    assert temp_case["persisted"] is False
    assert json.loads(temp_case["result_json"])["classification"]["predicted_class"] == "mel"
    assert "Bleeding" in temp_case["symptoms"]


def test_generate_report_export_content():
    """
    Verify generate_report_export produces structured Markdown containing:
    - patient and lesion info
    - ABCDE quantitative features
    - AI predictions and confidence
    - prominent warning notice indicating data is NOT saved to database
    - no fabricated findings
    """
    patient = {"full_name": "Bob Taylor", "age": 55, "gender": "Male", "phone": "987-654", "visit_date": "2026-10-16"}
    lesion = {"area": "Chest", "size": "7 mm", "duration": "6 months", "symptoms": ["Color change"], "explanation": "Darkened mole"}
    result = {
        "classification": {
            "predicted_class": "mel",
            "melanoma_probability": 0.88,
            "risk_level": "high",
        },
        "features": {
            "asymmetry": 0.65,
            "border_irregularity": 0.72,
            "color_index": 0.45,
            "diameter_mm": 7.4,
            "diameter_pixels": 220.0,
        },
        "evolution": {
            "available": True,
            "change_detected": True,
            "description": "Increase in diameter and asymmetry compared to baseline.",
        },
        "report": {
            "full_report_text": "Lesion exhibits marked structural asymmetry and multi-chromatic pattern.",
            "recommendation": "Urgent dermatological excision and histopathology recommended.",
        },
        "explanation": {
            "summary": "High risk features: diameter > 6mm and prominent border irregularities.",
        },
    }

    case = build_temp_case(patient, lesion, "lesion_bob.jpg", result)
    md_report = generate_report_export(case)

    # Patient info
    assert "Bob Taylor" in md_report
    assert "Chest" in md_report
    assert "55 yrs / Male" in md_report
    assert "Color change" in md_report

    # AI assessment
    assert "**Predicted Class:** mel" in md_report
    assert "88.0%" in md_report
    assert "**Overall Clinical Risk Level:** high" in md_report

    # ABCDE features
    assert "**Asymmetry (A):** 0.65" in md_report
    assert "**Border Irregularity (B):** 0.72" in md_report
    assert "**Color Variation (C):** 0.45" in md_report
    assert "**Diameter (D):** 7.40 mm" in md_report
    assert "WARNING: Exceeds 6.0 mm threshold" in md_report
    assert "**Evolution (E):** Significant change detected" in md_report

    # Structured report & recommendation
    assert "marked structural asymmetry" in md_report
    assert "Urgent dermatological excision" in md_report

    # Prominent storage notice
    assert "NOT SAVED TO SERVER DATABASE" in md_report
    assert "temporary session memory" in md_report.lower()
    assert "exported directly from temporary" in md_report.lower()


def test_sqlite_history_isolation_across_multiple_cases(tmp_path):
    """
    Verify across a series of 3 submissions (one persisted, one partial_success, one ambiguous)
    that ONLY the verified persisted case enters SQLite history.
    """
    db_path = tmp_path / "multi_cases.db"
    conn = create_test_db(db_path)

    cases_to_test = [
        # Case 1: Confirmed persistence -> should be saved
        (
            {"full_name": "Patient 1", "age": 20, "gender": "Male", "visit_date": "2026-10-16"},
            {"area": "Arm", "explanation": "Check"},
            {"status": "success", "persisted": True, "analysis_id": "id_1"}
        ),
        # Case 2: Partial success (MongoDB down) -> should NOT be saved
        (
            {"full_name": "Patient 2", "age": 30, "gender": "Female", "visit_date": "2026-10-16"},
            {"area": "Leg", "explanation": "Check"},
            {"status": "partial_success", "persisted": False, "storage_warning": "DB down"}
        ),
        # Case 3: Ambiguous (missing persisted field) -> should NOT be saved
        (
            {"full_name": "Patient 3", "age": 40, "gender": "Other", "visit_date": "2026-10-16"},
            {"area": "Back", "explanation": "Check"},
            {"status": "success"}
        ),
    ]

    for patient, lesion, result in cases_to_test:
        p_status, _ = classify_persistence_status(result)
        if should_save_to_history(p_status):
            case = build_temp_case(patient, lesion, "img.jpg", result)
            insert_case_record(conn, 1, case)

    rows = conn.execute("SELECT patient_name FROM cases").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "Patient 1"
    conn.close()
