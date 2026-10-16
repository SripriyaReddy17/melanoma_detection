import re


def validate_report(
    report,
    classification=None,
    abcde=None,
    evolution=None,
    rag_evidence=None,
):
    """
    Validate the generated CEFM report against the
    structured analysis results.

    This checker does not generate or modify the report.
    It only checks whether the report is consistent
    with the actual system outputs.
    """

    report = report or {}
    classification = classification or {}
    abcde = abcde or {}
    evolution = evolution or {}
    rag_evidence = rag_evidence or {}

    report_text = str(
        report.get("full_report_text", "")
    )

    checks = []

    # =========================================================
    # CLASSIFICATION CHECK
    # =========================================================

    predicted_class = str(
        classification.get(
            "predicted_class",
            "not available",
        )
    ).replace("_", " ")

    probability = _number(
        classification.get(
            "melanoma_probability"
        ),
        0.0,
    )

    classification_ok = True

    if predicted_class != "not available":
        if predicted_class.lower() not in report_text.lower():
            classification_ok = False

    probability_percent = round(
        probability * 100
    )

    if probability_percent > 0:
        probability_patterns = [
            f"{probability_percent}%",
            f"{probability_percent} %",
        ]

        if not any(
            pattern in report_text
            for pattern in probability_patterns
        ):
            classification_ok = False

    checks.append(
        _check(
            "classification_consistency",
            classification_ok,
            "Classification prediction and melanoma probability "
            "are consistent with the structured model output.",
        )
    )

    # =========================================================
    # A — ASYMMETRY
    # =========================================================

    asymmetry = _number(
        abcde.get("asymmetry"),
        None,
    )

    asymmetry_ok = _check_numeric_value(
        report_text,
        asymmetry,
        decimals=2,
    )

    checks.append(
        _check(
            "asymmetry_consistency",
            asymmetry_ok,
            "Reported asymmetry matches the extracted ABCDE value.",
        )
    )

    # =========================================================
    # B — BORDER
    # =========================================================

    border = _number(
        abcde.get("border_irregularity"),
        None,
    )

    border_ok = _check_numeric_value(
        report_text,
        border,
        decimals=2,
    )

    checks.append(
        _check(
            "border_consistency",
            border_ok,
            "Reported border irregularity matches the extracted "
            "ABCDE value.",
        )
    )

    # =========================================================
    # C — COLOR
    # =========================================================

    color_index = abcde.get(
        "color_index"
    )

    if color_index is None:
        color = abcde.get(
            "color",
            {}
        ) or {}

        color_index = color.get(
            "index"
        )

    color_index = _number(
        color_index,
        None,
    )

    color_ok = _check_numeric_value(
        report_text,
        color_index,
        decimals=2,
    )

    checks.append(
        _check(
            "color_consistency",
            color_ok,
            "Reported color variation matches the extracted "
            "ABCDE value.",
        )
    )

    # =========================================================
    # D — DIAMETER
    # =========================================================

    diameter_mm = abcde.get(
        "diameter_mm"
    )

    if diameter_mm is None:
        diameter = abcde.get(
            "diameter",
            {}
        ) or {}

        diameter_mm = diameter.get(
            "max_diameter_mm"
        )

    diameter_mm = _number(
        diameter_mm,
        None,
    )

    if diameter_mm is not None:
        diameter_ok = _check_numeric_value(
            report_text,
            diameter_mm,
            decimals=1,
        )

        expected_exceeds = (
            diameter_mm > 6
        )

        if expected_exceeds:
            diameter_ok = (
                diameter_ok
                and (
                    "above" in report_text.lower()
                    or "exceeds" in report_text.lower()
                )
            )
    else:
        diameter_ok = True

    checks.append(
        _check(
            "diameter_consistency",
            diameter_ok,
            "Reported diameter matches the calibrated measurement "
            "and its 6 mm comparison.",
        )
    )

    # =========================================================
    # E — EVOLUTION
    # =========================================================

    evolution_available = bool(
        evolution.get("available", False)
    )

    evolution_ok = True

    if not evolution_available:
        unavailable_phrases = [
            "previous image",
            "not available",
            "no previous",
        ]

        evolution_ok = any(
            phrase in report_text.lower()
            for phrase in unavailable_phrases
        )

    else:
        if "previous uploaded image" not in report_text.lower():
            evolution_ok = False

    checks.append(
        _check(
            "evolution_consistency",
            evolution_ok,
            "Evolution statement matches the availability of "
            "longitudinal comparison data.",
        )
    )

    # =========================================================
    # RAG CHECK
    # =========================================================

    rag_documents = (
        rag_evidence.get(
            "documents",
            []
        ) or []
    )

    rag_ok = True

    if rag_documents:
        if "medical evidence from rag" not in report_text.lower():
            rag_ok = False

        for index, document in enumerate(
            rag_documents,
            start=1,
        ):
            source = str(
                (document or {}).get(
                    "source",
                    ""
                )
            ).strip()

            if source and source.lower() not in report_text.lower():
                rag_ok = False

    else:
        if (
            "no relevant medical reference"
            not in report_text.lower()
        ):
            rag_ok = False

    checks.append(
        _check(
            "rag_evidence_consistency",
            rag_ok,
            "RAG evidence shown in the report matches the "
            "retrieved evidence.",
        )
    )

    # =========================================================
    # SAFETY CHECK
    # =========================================================

    safety_phrases = [
        "not a diagnosis",
        "cannot confirm or rule out melanoma",
        "qualified healthcare professional",
    ]

    safety_ok = all(
        phrase in report_text.lower()
        for phrase in safety_phrases
    )

    checks.append(
        _check(
            "medical_safety",
            safety_ok,
            "The report contains the required medical safety "
            "and limitation statements.",
        )
    )

    # =========================================================
    # UNSUPPORTED CERTAINTY CHECK
    # =========================================================

    unsafe_phrases = [
        "you have melanoma",
        "you do not have melanoma",
        "confirmed melanoma",
        "definitely melanoma",
    ]

    unsupported_claims = [
        phrase
        for phrase in unsafe_phrases
        if phrase in report_text.lower()
    ]

    certainty_ok = len(
        unsupported_claims
    ) == 0

    checks.append(
        _check(
            "unsupported_diagnostic_claims",
            certainty_ok,
            "The report does not make unsupported definitive "
            "diagnostic claims.",
        )
    )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    passed = sum(
        1
        for item in checks
        if item["passed"]
    )

    total = len(checks)

    all_passed = (
        passed == total
    )

    return {
        "status": (
            "passed"
            if all_passed
            else "failed"
        ),
        "fact_check_passed": all_passed,
        "checks_passed": passed,
        "checks_total": total,
        "checks": checks,
        "unsupported_claims": unsupported_claims,
    }


# =============================================================
# HELPERS
# =============================================================

def _number(value, default=0.0):
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _check_numeric_value(
    text,
    value,
    decimals=2,
):
    """
    Check whether a rounded numeric value appears
    in the generated report.
    """

    if value is None:
        return True

    rounded_value = round(
        float(value),
        decimals,
    )

    formatted = f"{rounded_value:.{decimals}f}"

    if formatted in text:
        return True

    # Also allow values without trailing zeroes.
    short_form = str(
        round(
            float(value),
            decimals,
        )
    )

    return short_form in text


def _check(
    name,
    passed,
    message,
):
    """
    Create a single fact-check result.
    """

    return {
        "name": name,
        "passed": bool(passed),
        "message": message,
    }