import numpy as np

from app.services import analysis
from app.services.report_generation import generate_structured_report


def _run_analysis_with_rag(monkeypatch, agent_result=None, agent_error=None, conventional_result=None, conventional_error=None):
    monkeypatch.setattr(
        analysis,
        "validate_image",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        analysis,
        "load_image",
        lambda *args, **kwargs: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        analysis,
        "segment_lesion",
        lambda *args, **kwargs: np.ones((8, 8), dtype=np.uint8),
    )
    monkeypatch.setattr(analysis, "_save_segmentation_overlay", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        analysis.image_type_detector,
        "predict",
        lambda *args, **kwargs: {"type": "dermoscopic"},
    )
    monkeypatch.setattr(
        analysis,
        "estimate_pixels_per_mm",
        lambda *args, **kwargs: {
            "valid": True,
            "pixels_per_mm": 1.0,
        },
    )
    monkeypatch.setattr(
        analysis,
        "check_post_segmentation_quality",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        analysis,
        "extract_abcde_features",
        lambda *args, **kwargs: {
            "asymmetry": 0.2,
            "border_irregularity": 0.3,
            "color": {"h": 0.1, "s": 0.2, "v": 0.3, "index": 0.4},
            "diameter_pixels": 4.0,
            "diameter": {
                "max_diameter_mm": 4.0,
                "available": True,
            },
        },
    )
    monkeypatch.setattr(
        analysis,
        "classify",
        lambda *args, **kwargs: {
            "predicted_class": "nv",
            "melanoma_probability": 0.1,
            "risk_level": "low",
        },
    )
    monkeypatch.setattr(
        analysis,
        "compute_evolution",
        lambda *args, **kwargs: {"available": False},
    )
    monkeypatch.setattr(
        analysis,
        "compute_abcde_score",
        lambda *args, **kwargs: {"score": 0.2, "label": "low"},
    )

    import app.services.multimodal as multimodal

    monkeypatch.setattr(
        multimodal,
        "run_multimodal_pipeline",
        lambda **kwargs: {"contrastive_alignment": {}},
    )

    class FakeClinicalAgent:
        def run(self, **kwargs):
            if agent_error is not None:
                raise agent_error
            return agent_result

    monkeypatch.setattr(analysis, "ClinicalRAGAgent", FakeClinicalAgent)

    def fake_conventional_rag(**kwargs):
        if conventional_error is not None:
            raise conventional_error
        return conventional_result

    monkeypatch.setattr(
        analysis,
        "retrieve_melanoma_evidence",
        fake_conventional_rag,
    )
    monkeypatch.setattr(
        analysis,
        "generate_structured_report",
        generate_structured_report,
    )
    monkeypatch.setattr(
        analysis,
        "validate_report",
        lambda **kwargs: {"valid": True},
    )

    return analysis.analyze_image(
        image_path="sample.jpg",
        original_filename="sample.jpg",
    )


def test_successful_agentic_rag_is_marked_agentic(monkeypatch):
    evidence = {
        "query": "clinical melanoma guidance",
        "documents": [{"source": "guideline.pdf", "text": "Guidance.", "score": 0.8}],
        "agent_trace": {"rounds": []},
        "provenance": [{"source": "guideline.pdf"}],
        "rounds_executed": 1,
        "sufficiency": "sufficient",
    }

    result = _run_analysis_with_rag(
        monkeypatch,
        agent_result=evidence,
    )

    assert result["rag_evidence"]["retrieval_mode"] == "agentic_rag"
    assert result["rag_evidence"]["agentic_rag"] is True
    assert result["rag_evidence"]["documents"] == evidence["documents"]
    assert result["rag_evidence"]["agent_trace"] == evidence["agent_trace"]
    assert result["agentic_rag"]["retrieval_mode"] == "agentic_rag"
    assert result["agentic_rag"]["agentic_rag"] is True
    assert "conventional RAG fallback" not in result["report"]["full_report_text"]


def test_agentic_failure_preserves_conventional_evidence_and_marks_fallback(monkeypatch):
    evidence = {
        "query": "conventional query",
        "documents": [{"source": "guideline.pdf", "text": "Guidance.", "score": 0.7}],
    }

    result = _run_analysis_with_rag(
        monkeypatch,
        agent_error=RuntimeError("private exception details"),
        conventional_result=evidence,
    )

    rag_evidence = result["rag_evidence"]
    assert rag_evidence["retrieval_mode"] == "conventional_rag_fallback"
    assert rag_evidence["agentic_rag"] is False
    assert rag_evidence["agentic_rag_error"] == "Agentic RAG failed (RuntimeError)."
    assert rag_evidence["documents"] == evidence["documents"]
    assert "private exception details" not in str(result["report"])
    assert (
        "Evidence retrieval mode: conventional RAG fallback because "
        "Agentic RAG was unavailable."
    ) in result["report"]["full_report_text"]


def test_both_retrieval_paths_failing_returns_no_fabricated_evidence(monkeypatch):
    result = _run_analysis_with_rag(
        monkeypatch,
        agent_error=RuntimeError("private agent details"),
        conventional_error=ConnectionError("private database details"),
    )

    rag_evidence = result["rag_evidence"]
    assert rag_evidence["retrieval_mode"] == "retrieval_failed"
    assert rag_evidence["agentic_rag"] is False
    assert rag_evidence["agentic_rag_error"] == "Agentic RAG failed (RuntimeError)."
    assert rag_evidence["documents"] == []
    assert rag_evidence["error"] == "Conventional RAG failed (ConnectionError)."
    assert "private agent details" not in str(result["report"])
    assert "private database details" not in str(result["report"])
