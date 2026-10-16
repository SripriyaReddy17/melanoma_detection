from app.services.contrastive import contrastive_aligner
from app.services.explainability import generate_explanation


def test_contrastive_aligner_returns_similarity_score():
    score = contrastive_aligner.compute_similarity(
        {"texture": 0.8, "color": 0.9},
        {"asymmetry": 0.7, "border_irregularity": 0.8, "color_index": 0.9},
    )

    assert 0.0 <= score <= 1.0
    assert isinstance(score, float)


def test_explainability_summary_mentions_abcde_signals():
    result = generate_explanation(
        {"melanoma_probability": 0.72, "risk_level": "high"},
        {"asymmetry": 0.7, "border_irregularity": 0.8, "color": {"index": 0.65}},
        {"change_detected": True},
    )

    summary = result["summary"].lower()
    assert "asymmetry" in summary
    assert "border" in summary or "irregularity" in summary
    assert "color" in summary
    assert "high" in summary or "risk" in summary
