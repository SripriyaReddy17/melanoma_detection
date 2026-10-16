"""
Integration and unit tests for FastAPI backend routes, configuration,
secret masking, and safe MongoDB failure handling in CEFM.
"""

import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import router
from app.core import config
from app.core.config import (
    Settings,
    sanitize_mongodb_url,
    sanitize_log_message,
    MODEL_RUNTIME_STATUS,
)
from app.main import app


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def test_image_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(180, 90, 40))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_pipeline_success(monkeypatch):
    mock = MagicMock(return_value={
        "status": "success",
        "validation": {"valid": True, "reason": "All basic image quality checks passed."},
        "image_type": {"image_type": "dermoscopic", "supported": True},
        "calibration": {"calibrated": True, "pixels_per_mm": 10.0},
        "features": {
            "asymmetry": 0.25,
            "border_irregularity": 0.35,
            "color_h": 12.0,
            "color_s": 22.0,
            "color_v": 32.0,
            "color_index": 0.28,
            "diameter_pixels": 150.0,
            "diameter_mm": 15.0,
        },
        "classification": {
            "predicted_class": "non_mel",
            "melanoma_probability": 0.22,
            "risk_level": "low",
        },
        "evolution": {"available": False, "change_detected": False},
        "multimodal": {},
        "contrastive_alignment": {},
        "explanation": {"summary": "Lesion exhibits low asymmetry and regular border."},
        "rag_evidence": {"retrieval_rounds": 1},
        "report": {"full_report_text": "Clinical findings suggest a benign nevus."},
    })
    monkeypatch.setattr("app.api.routes.analyze_image", mock)
    return mock


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """Sets up an isolated environment with mocked database collections and temp upload dir."""
    monkeypatch.setattr("app.api.routes.UPLOAD_DIR", tmp_path)

    mock_cases = MagicMock()
    mock_analyses = MagicMock()

    monkeypatch.setattr("app.api.routes.cases_collection", mock_cases)
    monkeypatch.setattr("app.api.routes.analyses_collection", mock_analyses)

    with TestClient(app) as client:
        yield client, mock_cases, mock_analyses, tmp_path


# ============================================================
# TASK A: CONFIGURATION & SECRET MASKING TESTS
# ============================================================

def test_config_defaults():
    s = Settings()
    assert s.mongodb_url == "mongodb://localhost:27017"
    assert s.mongodb_database == "cefm_melanoma"
    assert s.max_upload_size_mb == 25
    assert s.max_upload_bytes == 25 * 1024 * 1024


def test_config_environment_overrides(monkeypatch):
    monkeypatch.setenv("MONGODB_URL", "mongodb://dbserver.internal:27017")
    monkeypatch.setenv("MONGODB_DATABASE", "custom_cefm_db")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "50")

    s = Settings()
    assert s.mongodb_url == "mongodb://dbserver.internal:27017"
    assert s.mongodb_database == "custom_cefm_db"
    assert s.max_upload_size_mb == 50
    assert s.max_upload_bytes == 50 * 1024 * 1024


def test_secret_masking_in_urls_and_logs():
    raw_url = "mongodb://clinical_user:SuperSecretPassword123@mongo.hospital.org:27017/cefm_melanoma?authSource=admin"
    masked = sanitize_mongodb_url(raw_url)

    assert "SuperSecretPassword123" not in masked
    assert "clinical_user:***@" in masked
    assert "mongo.hospital.org:27017" in masked

    raw_err = f"Failed to connect to {raw_url}: ServerSelectionTimeoutError"
    masked_err = sanitize_log_message(raw_err)
    assert "SuperSecretPassword123" not in masked_err
    assert "clinical_user:***@" in masked_err


def test_model_runtime_status_does_not_expose_secrets():
    deepseek_status = MODEL_RUNTIME_STATUS.get("deepseek", {})
    # The API key value itself must not be exposed in runtime status dict
    assert "api_key" not in deepseek_status
    assert "configured" in deepseek_status


# ============================================================
# HEALTH CHECK TESTS
# ============================================================

def test_health_check_healthy(test_env, monkeypatch):
    client, _, _, _ = test_env
    monkeypatch.setattr("app.database.check_mongodb", lambda: True)

    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["mongodb"] == "connected"


def test_health_check_degraded_when_mongodb_down(test_env, monkeypatch):
    client, _, _, _ = test_env
    monkeypatch.setattr("app.database.check_mongodb", lambda: False)

    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.json()["mongodb"] == "unreachable"


# ============================================================
# TASK C: SAFE PERSISTENCE FAILURE HANDLING IN /analyze
# ============================================================

def test_analyze_successful_persistence(test_env, test_image_bytes, mock_pipeline_success):
    client, mock_cases, mock_analyses, upload_dir = test_env

    mock_cases.find_one.return_value = {"_id": "case_abc", "label": "Patient Jane"}
    mock_analyses.find_one.return_value = None
    mock_analyses.insert_one.return_value = None

    files = {"image": ("lesion.jpg", test_image_bytes, "image/jpeg")}
    data = {"patient_name": "Patient Jane"}

    resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["persisted"] is True
    assert payload["storage_status"] == "saved"
    assert payload["analysis_id"] is not None
    assert payload["case_id"] == "case_abc"
    assert payload["image_url"].startswith("/api/v1/images/")
    assert mock_analyses.insert_one.called


def test_analyze_persistence_failure_returns_real_results_with_warning(
    test_env, test_image_bytes, mock_pipeline_success
):
    client, mock_cases, mock_analyses, upload_dir = test_env

    mock_cases.find_one.return_value = {"_id": "case_abc", "label": "Patient Jane"}
    mock_analyses.find_one.return_value = None
    # Simulate MongoDB write failure
    mock_analyses.insert_one.side_effect = RuntimeError("MongoDB connection dropped during insert")

    files = {"image": ("lesion.jpg", test_image_bytes, "image/jpeg")}
    data = {"patient_name": "Patient Jane"}

    resp = client.post("/api/v1/analyze", files=files, data=data)

    # Must retain 200 with partial_success without throwing 500 or fake IDs
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "partial_success"
    assert payload["persisted"] is False
    assert payload["storage_status"] == "persistence_failed"
    assert payload["analysis_id"] is None  # Must NOT supply a misleading fake DB ID
    assert payload["image_url"] is None
    assert "storage_warning" in payload
    assert "could not be persisted" in payload["storage_warning"].lower()

    # Clinical findings must NOT be discarded
    assert payload["classification"]["predicted_class"] == "non_mel"
    assert payload["abcde"]["asymmetry"] == 0.25
    assert payload["report"]["full_report_text"] == "Clinical findings suggest a benign nevus."


def test_analyze_case_resolution_db_failure_proceeds_with_unpersisted_result(
    test_env, test_image_bytes, mock_pipeline_success
):
    client, mock_cases, mock_analyses, upload_dir = test_env

    # Simulate database down during case lookup/creation
    mock_cases.find_one.side_effect = RuntimeError("MongoDB down")
    mock_cases.insert_one.side_effect = RuntimeError("MongoDB down")

    files = {"image": ("lesion.jpg", test_image_bytes, "image/jpeg")}
    data = {"patient_name": "Patient Jane"}

    resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "partial_success"
    assert payload["persisted"] is False
    assert payload["storage_status"] == "persistence_failed"
    assert payload["case_id"] is None
    assert payload["analysis_id"] is None
    assert payload["classification"]["predicted_class"] == "non_mel"


def test_analyze_calibration_rejection_surfaced_with_400(test_env, test_image_bytes, monkeypatch):
    client, mock_cases, mock_analyses, upload_dir = test_env

    mock_cases.find_one.return_value = {"_id": "case_1", "label": "Case 1"}
    monkeypatch.setattr("app.api.routes.analyze_image", lambda *a, **kw: {
        "status": "invalid",
        "validation": {
            "valid": False,
            "calibration_rejection": {"message": "Physical calibration reference not detected."},
            "reason": "Calibration failed.",
        },
        "rejection_stage": "calibration",
    })

    files = {"image": ("lesion.jpg", test_image_bytes, "image/jpeg")}
    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["message"] == "Image validation failed."
    assert "Physical calibration reference not detected" in detail["reason"]
    # No analysis record must be saved in MongoDB
    assert not mock_analyses.insert_one.called


def test_analyze_pipeline_exception_cleans_up_and_returns_500(test_env, test_image_bytes, monkeypatch):
    client, mock_cases, mock_analyses, upload_dir = test_env

    mock_cases.find_one.return_value = {"_id": "case_1", "label": "Case 1"}

    def explode(*a, **kw):
        raise ValueError("Model weight failure in UltraLight VM-UNet")

    monkeypatch.setattr("app.api.routes.analyze_image", explode)

    files = {"image": ("lesion.jpg", test_image_bytes, "image/jpeg")}
    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 500
    assert "Analysis pipeline encountered an error" in resp.json()["detail"]
    assert not mock_analyses.insert_one.called
    assert list(upload_dir.glob("*")) == []


# ============================================================
# CASE ROUTES TESTS
# ============================================================

def test_create_case_success(test_env):
    client, mock_cases, _, _ = test_env
    mock_cases.insert_one.return_value = None

    resp = client.post("/api/v1/cases", data={"label": "Patient Alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "Patient Alpha"
    assert "case_id" in data


def test_create_case_db_failure_returns_503(test_env):
    client, mock_cases, _, _ = test_env
    mock_cases.insert_one.side_effect = RuntimeError("Database unreachable")

    resp = client.post("/api/v1/cases", data={"label": "Patient Alpha"})
    assert resp.status_code == 503
    assert "database service is temporarily unavailable" in resp.json()["detail"].lower()


def test_list_cases_success(test_env):
    client, mock_cases, mock_analyses, _ = test_env
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {"_id": "c1", "label": "Patient 1", "created_at": datetime(2026, 1, 1)},
        {"_id": "c2", "label": "Patient 2", "created_at": datetime(2026, 1, 2)},
    ]
    mock_cases.find.return_value = mock_cursor
    mock_analyses.count_documents.return_value = 2

    resp = client.get("/api/v1/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["case_id"] == "c1"
    assert data[0]["num_analyses"] == 2


def test_list_cases_db_failure_returns_503(test_env):
    client, mock_cases, _, _ = test_env
    mock_cases.find.side_effect = RuntimeError("Mongo down")

    resp = client.get("/api/v1/cases")
    assert resp.status_code == 503
    assert "database service is temporarily unavailable" in resp.json()["detail"].lower()


def test_get_case_success(test_env):
    client, mock_cases, mock_analyses, _ = test_env
    mock_cases.find_one.return_value = {
        "_id": "c1",
        "label": "Patient 1",
        "created_at": datetime(2026, 1, 1),
    }
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = [
        {
            "_id": "a1",
            "created_at": datetime(2026, 1, 1),
            "predicted_class": "non_mel",
            "melanoma_probability": 0.1,
            "risk_level": "low",
        }
    ]
    mock_analyses.find.return_value = mock_cursor

    resp = client.get("/api/v1/cases/c1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["case_id"] == "c1"
    assert len(data["analyses"]) == 1
    assert data["analyses"][0]["analysis_id"] == "a1"


def test_get_case_not_found_returns_404(test_env):
    client, mock_cases, _, _ = test_env
    mock_cases.find_one.return_value = None

    resp = client.get("/api/v1/cases/missing_c")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Case not found."


def test_get_case_db_failure_returns_503(test_env):
    client, mock_cases, _, _ = test_env
    mock_cases.find_one.side_effect = RuntimeError("Mongo timeout")

    resp = client.get("/api/v1/cases/c1")
    assert resp.status_code == 503


# ============================================================
# ANALYSIS ROUTES TESTS
# ============================================================

def test_list_analyses_success(test_env):
    client, _, mock_analyses, _ = test_env
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = [
        {
            "_id": "a1",
            "case_id": "c1",
            "original_filename": "lesion.jpg",
            "created_at": datetime(2026, 1, 1),
            "predicted_class": "mel",
            "melanoma_probability": 0.85,
            "risk_level": "high",
        }
    ]
    mock_analyses.find.return_value = mock_cursor

    resp = client.get("/api/v1/analyses?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["analysis_id"] == "a1"
    assert data[0]["predicted_class"] == "mel"


def test_list_analyses_db_failure_returns_503(test_env):
    client, _, mock_analyses, _ = test_env
    mock_analyses.find.side_effect = RuntimeError("Mongo failure")

    resp = client.get("/api/v1/analyses")
    assert resp.status_code == 503


def test_get_analysis_success(test_env):
    client, _, mock_analyses, _ = test_env
    mock_analyses.find_one.return_value = {
        "_id": "a1",
        "case_id": "c1",
        "original_filename": "lesion.jpg",
        "created_at": datetime(2026, 1, 1),
        "asymmetry": 0.3,
        "border_irregularity": 0.4,
        "predicted_class": "non_mel",
        "melanoma_probability": 0.12,
        "risk_level": "low",
        "report": {"full_report_text": "Benign nevus."},
    }

    resp = client.get("/api/v1/analyses/a1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_id"] == "a1"
    assert data["abcde"]["asymmetry"] == 0.3
    assert data["classification"]["predicted_class"] == "non_mel"


def test_get_analysis_not_found_returns_404(test_env):
    client, _, mock_analyses, _ = test_env
    mock_analyses.find_one.return_value = None

    resp = client.get("/api/v1/analyses/missing_id")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Analysis not found."


def test_get_analysis_db_failure_returns_503(test_env):
    client, _, mock_analyses, _ = test_env
    mock_analyses.find_one.side_effect = RuntimeError("Mongo failure")

    resp = client.get("/api/v1/analyses/a1")
    assert resp.status_code == 503


# ============================================================
# IMAGE & SEGMENTATION RETRIEVAL TESTS
# ============================================================

def test_get_image_file_success(test_env, tmp_path):
    client, _, mock_analyses, _ = test_env
    img_file = tmp_path / "img_a1.jpg"
    img_file.write_bytes(b"dummy_image_data")

    mock_analyses.find_one.return_value = {
        "_id": "a1",
        "stored_path": str(img_file),
    }

    resp = client.get("/api/v1/images/a1")
    assert resp.status_code == 200
    assert resp.content == b"dummy_image_data"


def test_get_image_file_not_found_404(test_env):
    client, _, mock_analyses, _ = test_env
    mock_analyses.find_one.return_value = {
        "_id": "a1",
        "stored_path": "non_existent_file.jpg",
    }

    resp = client.get("/api/v1/images/a1")
    assert resp.status_code == 404


def test_get_segmentation_image_success(test_env, tmp_path):
    client, _, mock_analyses, _ = test_env
    seg_file = tmp_path / "seg_a1.png"
    seg_file.write_bytes(b"\x89PNG\r\n\x1a\nfake_png")

    mock_analyses.find_one.return_value = {
        "_id": "a1",
        "segmentation_path": str(seg_file),
    }

    resp = client.get("/api/v1/images/a1/segmentation")
    assert resp.status_code == 200
    assert resp.content == b"\x89PNG\r\n\x1a\nfake_png"
