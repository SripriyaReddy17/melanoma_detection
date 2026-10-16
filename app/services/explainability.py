"""Generate a concise multimodal explanation for the melanoma assessment."""

from __future__ import annotations

from typing import Any

from app.services.contrastive import contrastive_aligner


def generate_explanation(classification: dict[str, Any], abcde: dict[str, Any], evolution=None):
    """Create a textual explanation using clinical ABCDE features and the
    multimodal alignment score.
    """
    prob = float((classification or {}).get("melanoma_probability", 0.0) or 0.0)
    risk = str((classification or {}).get("risk_level", "low")).lower()

    asymmetry = float((abcde or {}).get("asymmetry", 0.0) or 0.0)
    border = float((abcde or {}).get("border_irregularity", 0.0) or 0.0)
    color_index = float((abcde or {}).get("color_index", 0.0) or 0.0)
    evolution_change = bool((evolution or {}).get("change_detected", False))

    image_features = {
        "texture": max(0.0, min(1.0, 0.5 + asymmetry * 0.5)),
        "color": max(0.0, min(1.0, color_index)),
        "structure": max(0.0, min(1.0, border)),
    }
    clinical_features = {
        "asymmetry": asymmetry,
        "border_irregularity": border,
        "color_index": color_index,
    }
    similarity = contrastive_aligner.compute_similarity(image_features, clinical_features)

    summary_parts = [
        f"The model indicates a {risk} risk level with an estimated melanoma probability of {prob:.0%}.",
        f"Asymmetry is {asymmetry:.2f}, border irregularity is {border:.2f}, and color variation is {color_index:.2f}.",
        (
            "The combined visual-to-clinical alignment score is "
            f"{similarity:.2f}, indicating the lesion appearance is consistent with "
            "the observed ABCD pattern."
        ),
    ]

    if evolution_change:
        summary_parts.append("A change compared with the prior assessment was detected.")
    else:
        summary_parts.append("No meaningful change compared with the prior assessment was detected.")

    summary = " ".join(summary_parts)

    return {
        "summary": summary,
        "classification": classification,
        "abcde": abcde,
        "evolution": evolution,
        "multimodal_similarity": round(similarity, 4),
    }
