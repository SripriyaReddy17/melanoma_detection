"""
Tests for secure image upload limits and validation in CEFM FastAPI backend.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes import router
from app.core import config
from app.main import app


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(50, 150, 200))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_analyze_image(monkeypatch):
    mock = MagicMock(return_value={
        "status": "success",
        "validation": {"valid": True, "reason": "All checks passed."},
        "image_type": {"image_type": "dermoscopic", "supported": True},
        "calibration": {"calibrated": False},
        "features": {
            "asymmetry": 0.2,
            "border_irregularity": 0.3,
            "color_h": 10.0,
            "color_s": 20.0,
            "color_v": 30.0,
            "color_index": 0.25,
            "diameter_pixels": 120.0,
            "diameter_mm": None,
        },
        "classification": {
            "predicted_class": "non_mel",
            "melanoma_probability": 0.15,
            "risk_level": "low",
        },
        "evolution": {"available": False},
        "multimodal": {},
        "contrastive_alignment": {},
        "explanation": {"summary": "Low risk lesion."},
        "rag_evidence": {},
        "report": {"full_report_text": "Benign appearance."},
    })
    monkeypatch.setattr("app.api.routes.analyze_image", mock)
    return mock


@pytest.fixture
def client_with_tmp_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.routes.UPLOAD_DIR", tmp_path)
    # Mock MongoDB collections so tests never touch production MongoDB
    mock_cases = MagicMock()
    mock_cases.find_one.return_value = {"_id": "case_123", "label": "Test Case"}
    mock_cases.insert_one.return_value = None

    mock_analyses = MagicMock()
    mock_analyses.find_one.return_value = None
    mock_analyses.insert_one.return_value = None

    monkeypatch.setattr("app.api.routes.cases_collection", mock_cases)
    monkeypatch.setattr("app.api.routes.analyses_collection", mock_analyses)

    with TestClient(app) as client:
        yield client, tmp_path


def test_valid_jpeg_within_limit(client_with_tmp_upload, sample_jpeg_bytes, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    files = {"image": ("test_lesion.jpg", sample_jpeg_bytes, "image/jpeg")}
    data = {"patient_name": "Test Patient"}

    resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["filename"] == "test_lesion.jpg"
    assert mock_analyze_image.called


def test_valid_png_within_limit(client_with_tmp_upload, sample_png_bytes, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    files = {"image": ("test_lesion.png", sample_png_bytes, "image/png")}
    data = {"patient_name": "Test Patient"}

    resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert payload["filename"] == "test_lesion.png"
    assert mock_analyze_image.called


def test_oversized_image_rejected_413(client_with_tmp_upload, monkeypatch, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    # Set a tiny max upload limit of 1000 bytes for testing
    monkeypatch.setattr("app.api.routes.MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr("app.api.routes.MAX_UPLOAD_SIZE_MB", 1)

    # 5 KB payload
    big_bytes = b"\xff\xd8\xff" + b"0" * 5000
    files = {"image": ("big_image.jpg", big_bytes, "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 413
    assert "maximum allowed size" in resp.json()["detail"].lower()
    # Model inference must NOT be invoked on oversized uploads
    assert not mock_analyze_image.called
    # No temporary files should linger in upload_dir
    assert list(upload_dir.glob("*")) == []


def test_misleading_content_length_rejected(client_with_tmp_upload, monkeypatch, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    monkeypatch.setattr("app.api.routes.MAX_UPLOAD_BYTES", 500)
    monkeypatch.setattr("app.api.routes.MAX_UPLOAD_SIZE_MB", 1)

    # Payload larger than 500 bytes even if content-length header was smaller or missing
    oversized_data = b"\xff\xd8\xff" + b"A" * 1200
    files = {"image": ("test.jpg", oversized_data, "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 413
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_empty_upload_rejected_400(client_with_tmp_upload, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    files = {"image": ("empty.jpg", b"", "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_invalid_magic_bytes_rejected_400(client_with_tmp_upload, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    # Text file disguised as JPG
    fake_bytes = b"This is plain text pretending to be an image file."
    files = {"image": ("fake.jpg", fake_bytes, "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    assert "valid jpeg or png" in resp.json()["detail"].lower()
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_corrupted_image_data_rejected_400(client_with_tmp_upload, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    # Valid JPEG header followed by random corrupted garbage
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00CORRUPT_BYTES_TRUNCATED"
    files = {"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    assert "invalid or corrupted" in resp.json()["detail"].lower()
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_unsupported_content_type_rejected_400(client_with_tmp_upload, sample_jpeg_bytes, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    files = {"image": ("test.jpg", sample_jpeg_bytes, "text/plain")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    assert "supported" in resp.json()["detail"].lower()
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_unsupported_extension_rejected_400(client_with_tmp_upload, sample_jpeg_bytes, mock_analyze_image):
    client, upload_dir = client_with_tmp_upload
    files = {"image": ("test.gif", sample_jpeg_bytes, "image/jpeg")}

    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 400
    assert "extension" in resp.json()["detail"].lower()
    assert not mock_analyze_image.called
    assert list(upload_dir.glob("*")) == []


def test_temporary_file_cleaned_up_on_pipeline_error(client_with_tmp_upload, sample_jpeg_bytes, monkeypatch):
    client, upload_dir = client_with_tmp_upload

    def failing_pipeline(*args, **kwargs):
        raise RuntimeError("Unexpected pipeline crash")

    monkeypatch.setattr("app.api.routes.analyze_image", failing_pipeline)

    files = {"image": ("error.jpg", sample_jpeg_bytes, "image/jpeg")}
    resp = client.post("/api/v1/analyze", files=files)

    assert resp.status_code == 500
    # Must not leak stack traces or internal paths to client
    assert "Unexpected pipeline crash" not in resp.json()["detail"]
    assert "Analysis pipeline encountered an error" in resp.json()["detail"]
    # Disk cleanup check: file must be unlinked
    assert list(upload_dir.glob("*")) == []
