def generate_structured_report(
    validation,
    classification=None,
    abcde=None,
    evolution=None,
    explanation=None,
    rag_evidence=None,
):
    """
    Generate the structured melanoma assessment report.

    RAG provides supporting medical reference information.
    It does not change or override the model prediction.
    """

    # =========================================================
    # NORMALIZE INPUTS
    # =========================================================

    validation = validation or {}
    classification = classification or {}
    abcde = abcde or {}
    evolution = evolution or {}
    explanation = explanation or {}
    rag_evidence = rag_evidence or {}

    # =========================================================
    # RAG DOCUMENTS
    # =========================================================

    rag_documents = rag_evidence.get(
        "documents",
        []
    ) or []

    # =========================================================
    # ABCDE DATA
    # =========================================================

    # Current analysis.py uses flattened ABCDE fields.
    # Nested fallback is retained for backward compatibility.

    color = abcde.get(
        "color",
        {}
    ) or {}

    color_index = abcde.get(
        "color_index",
        color.get("index", 0.0)
    )

    diameter = abcde.get(
        "diameter",
        {}
    ) or {}

    diameter_mm = abcde.get(
        "diameter_mm",
        diameter.get("max_diameter_mm")
    )

    diameter_pixels = abcde.get(
        "diameter_pixels",
        diameter.get("max_diameter_pixels")
    )

    diameter_available = abcde.get(
        "diameter_mm_available",
        diameter.get("available", False)
    )

    # =========================================================
    # CLASSIFICATION DATA
    # =========================================================

    probability = _number(
        classification.get(
            "melanoma_probability"
        ),
        0.0,
    )

    risk = str(
        classification.get(
            "risk_level",
            "not available",
        )
    ).replace("_", " ")

    predicted_class = str(
        classification.get(
            "predicted_class",
            "not available",
        )
    ).replace("_", " ")

    # =========================================================
    # DIAMETER
    # =========================================================

    exceeds_6mm = (
        diameter_mm is not None
        and _number(diameter_mm) > 6
    )

    if (
        diameter_available
        and diameter_mm is not None
    ):
        diameter_text = (
            f"The measured maximum diameter is "
            f"{_number(diameter_mm):.1f} mm. "
            f"{'This is above' if exceeds_6mm else 'This is not above'} "
            "the conventional 6 mm reference "
            "used in ABCDE screening."
        )

    elif diameter_pixels is not None:
        diameter_text = (
            f"The lesion measures approximately "
            f"{_number(diameter_pixels):.0f} pixels "
            "in the image. A physical millimetre "
            "measurement was not available because "
            "the calibration reference could not "
            "be validated."
        )

    else:
        diameter_text = (
            "A reliable diameter measurement "
            "was not available."
        )

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    validation_text = (
        "The image passed the available "
        "image-quality checks."
    )

    if not validation.get("valid", True):
        validation_text = (
            "The image did not pass all "
            "image-quality checks."
        )

    # =========================================================
    # EVOLUTION
    # =========================================================

    evolution_text = _evolution_text(
        evolution
    )

    # =========================================================
    # MODEL EXPLANATION
    # =========================================================

    summary = explanation.get(
        "summary",
        "",
    )

    # =========================================================
    # RAG MEDICAL EVIDENCE
    # =========================================================

    rag_text = _format_rag_evidence(
        rag_documents
    )

    retrieval_mode_text = (
        "Evidence retrieval mode: conventional RAG fallback because "
        "Agentic RAG was unavailable."
        if rag_evidence.get("retrieval_mode")
        == "conventional_rag_fallback"
        else None
    )

    # =========================================================
    # FULL REPORT
    # =========================================================

    full_report_text = "\n\n".join(
        [
            "CEFM Melanoma Assessment Report",

            "What this report says",

            (
                f"The image analysis estimates a "
                f"{risk} level of concern. The computer "
                f"classified the image as "
                f"'{predicted_class}' with an estimated "
                f"melanoma probability of "
                f"{probability:.0%}. This is an "
                "image-based estimate, not a diagnosis."
            ),

            "ABCDE findings",

            (
                f"A - Asymmetry: "
                f"{_number(abcde.get('asymmetry')):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "the two sides of the measured lesion "
                "look less alike."
            ),

            (
                f"B - Border: "
                f"{_number(abcde.get('border_irregularity')):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "the outline appears more irregular."
            ),

            (
                f"C - Color variation: "
                f"{_number(color_index):.2f} "
                "on a 0 to 1 scale. Higher values mean "
                "more variation in the colors inside "
                "the lesion."
            ),

            f"D - Diameter: {diameter_text}",

            f"E - Evolution: {evolution_text}",

            f"Image quality: {validation_text}",

            (
                f"Overall explanation: {summary}"
                if summary
                else
                "Overall explanation: The available "
                "measurements are shown above."
            ),

            # =================================================
            # RAG SECTION
            # =================================================

            *(
                [retrieval_mode_text]
                if retrieval_mode_text
                else []
            ),
            "Medical evidence from RAG",

            rag_text,

            # =================================================
            # RECOMMENDATION
            # =================================================

            (
                "What to do next: This result should be "
                "reviewed by a qualified healthcare "
                "professional, especially if the lesion "
                "is changing, bleeding, itching, painful, "
                "or looks different from your other moles."
            ),

            # =================================================
            # DISCLAIMER
            # =================================================

            (
                "Important: This tool supports research "
                "and clinical decision-making. It cannot "
                "confirm or rule out melanoma and should "
                "not replace an in-person medical "
                "examination."
            ),
        ]
    )

    # =========================================================
    # RETURN REPORT
    # =========================================================

    return {
        "validation": validation,
        "classification": classification,
        "abcde": abcde,
        "evolution": evolution,
        "explanation": explanation,

        # RAG information
        "rag_evidence": rag_evidence,

        # Complete report
        "full_report_text": full_report_text,

        "recommendation": (
            "This system is a research "
            "decision-support prototype. "
            "Clinical assessment must be performed "
            "by a qualified professional."
        ),
    }


# =============================================================
# RAG EVIDENCE FORMATTER
# =============================================================

def _format_rag_evidence(rag_documents):
    """
    Format retrieved RAG documents for the report.
    """

    if not rag_documents:
        return (
            "No relevant medical reference was "
            "retrieved from the RAG knowledge base."
        )

    sections = []

    for index, document in enumerate(
        rag_documents,
        start=1,
    ):
        document = document or {}

        source = document.get(
            "source",
            "Medical Reference",
        )

        text = document.get(
            "text",
            "",
        )

        score = _number(
            document.get("score"),
            0.0,
        )

        if not text:
            text = (
                "No text was available for "
                "this retrieved reference."
            )

        section = (
            f"{index}. {source}\n"
            f"Similarity score: {score:.2f}\n"
            f"{text}"
        )

        sections.append(section)

    return "\n\n".join(sections)


# =============================================================
# NUMBER HELPER
# =============================================================

def _number(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:
        return (
            float(value)
            if value is not None
            else default
        )

    except (TypeError, ValueError):
        return default


# =============================================================
# EVOLUTION TEXT
# =============================================================

def _evolution_text(evolution):
    """
    Generate human-readable evolution information.
    """

    if not evolution.get("available"):
        return evolution.get(
            "reason",
            "There is no previous image "
            "available for comparison.",
        )

    changes = [
        (key, item)
        for key, item in (
            evolution.get("changes") or {}
        ).items()
        if item.get("change_detected")
    ]

    if not changes:
        return (
            "The current image was compared "
            "with the previous uploaded image "
            "in the same case. No meaningful "
            "difference was detected in the "
            "measured features."
        )

    labels = {
        "diameter_mm": "diameter",
        "asymmetry": "asymmetry",
        "border_irregularity": "border shape",
        "color_h": "color hue",
        "color_s": "color saturation",
        "color_v": "color brightness",
        "color_index": "overall color variation",
    }

    changed_labels = ", ".join(
        labels.get(key, key)
        for key, _ in changes
    )

    return (
        "The current image was compared "
        "with the previous uploaded image "
        "in the same case. A meaningful "
        f"difference was detected in: "
        f"{changed_labels}. "

        "Differences can result from real "
        "change, lighting, camera angle, "
        "focus, or calibration, so a "
        "healthcare professional should "
        "review them."
    )