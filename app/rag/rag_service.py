from app.rag.retrieval import retrieve_documents


def build_melanoma_query(
    classification: dict,
    abcde: dict,
    evolution: dict,
) -> str:

    classification = classification or {}
    abcde = abcde or {}
    evolution = evolution or {}

    prediction = classification.get(
        "predicted_class",
        "unknown",
    )

    probability = classification.get(
        "melanoma_probability",
        0,
    )

    asymmetry = abcde.get(
        "asymmetry",
        0,
    )

    border = abcde.get(
        "border_irregularity",
        0,
    )

    color = abcde.get(
        "color_index",
        0,
    )

    diameter = abcde.get(
        "diameter_mm",
    )

    evolution_change = evolution.get(
        "change_detected",
        False,
    )

    query = f"""
Melanoma assessment and ABCDE clinical interpretation.

Predicted class: {prediction}

Model melanoma probability: {probability}

Asymmetry score: {asymmetry}

Border irregularity score: {border}

Color variation index: {color}

Diameter in millimetres: {diameter}

Evolution change detected: {evolution_change}

Explain the clinical significance of these melanoma-related features,
ABCDE criteria, warning signs, and appropriate professional follow-up.
"""

    return query.strip()


def retrieve_melanoma_evidence(
    classification: dict,
    abcde: dict,
    evolution: dict,
    top_k: int = 3,
):

    query = build_melanoma_query(
        classification=classification,
        abcde=abcde,
        evolution=evolution,
    )

    documents = retrieve_documents(
        query,
        top_k=top_k,
    )

    return {
        "query": query,
        "documents": documents,
    }