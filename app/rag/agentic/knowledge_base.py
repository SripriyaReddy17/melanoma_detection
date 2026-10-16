"""
Curated Medical Knowledge Base for Melanoma and ABCDE Assessment (Rule 29).
Includes structured metadata for clinical evidence retrieval.
"""

CURATED_KNOWLEDGE_DOCUMENTS = [
    {
        "id": "KB-MEL-001",
        "topic": "melanoma_overview",
        "feature": "classification",
        "source_type": "clinical_guideline",
        "year": 2024,
        "reliability": 0.98,
        "title": "Cutaneous Melanoma Clinical Presentation and Guidelines",
        "text": (
            "Cutaneous melanoma arises from malignant transformation of melanocytes. "
            "Early identification is critical because thin, localized lesions have high cure rates with surgical excision, "
            "whereas invasive or metastatic melanomas carry high mortality. Dermoscopy markedly improves diagnostic sensitivity "
            "and specificity compared with naked-eye examination. AI-assisted models provide supplementary triage but must be "
            "interpreted in correlation with full skin examination and dermatological histology."
        ),
    },
    {
        "id": "KB-BEN-002",
        "topic": "benign_lesions",
        "feature": "classification",
        "source_type": "textbook",
        "year": 2023,
        "reliability": 0.95,
        "title": "Benign Melanocytic Nevi Characteristics",
        "text": (
            "Benign melanocytic nevi typically present with geometric symmetry, regular smooth borders, uniform coloration, "
            "and structural stability over time. Common variants include junctional, compound, and intradermal nevi. Dysplastic "
            "or atypical nevi may exhibit mild asymmetry or border irregularity, warranting monitoring without necessarily indicating malignancy."
        ),
    },
    {
        "id": "KB-ASYM-003",
        "topic": "asymmetry_criteria",
        "feature": "A",
        "source_type": "peer_reviewed_paper",
        "year": 2024,
        "reliability": 0.96,
        "title": "Asymmetry in Melanoma Diagnosis: ABCD Criteria",
        "text": (
            "Asymmetry (A) is assessed along orthogonal axes bisecting the lesion. In benign nevi, the two halves typically mirror "
            "each other in contour and internal pigment distribution. High asymmetry indicates differential clonal expansion and "
            "uncontrolled growth characteristic of malignant melanoma. However, mechanical friction or trauma can occasionally produce "
            "secondary asymmetry in benign lesions."
        ),
    },
    {
        "id": "KB-BORD-004",
        "topic": "border_irregularity",
        "feature": "B",
        "source_type": "clinical_guideline",
        "year": 2024,
        "reliability": 0.96,
        "title": "Border Irregularity and Contour Notch Evaluation",
        "text": (
            "Border irregularity (B) reflects jagged, scalloped, notched, or abruptly terminating margins. In dermoscopy, abrupt cutoffs "
            "of the pigment network at the periphery or sharply defined angular corners correlate with rapid lateral radial growth phases. "
            "Quantified curvature variation and high boundary roughness are validated clinical indicators of atypical lesion behavior."
        ),
    },
    {
        "id": "KB-COL-005",
        "topic": "color_variation",
        "feature": "C",
        "source_type": "peer_reviewed_paper",
        "year": 2023,
        "reliability": 0.97,
        "title": "Color Variegation and Dermoscopic Chromatic Signs",
        "text": (
            "Color variation (C) evaluates heterogeneity of pigmentation. Presence of multiple colors—including shades of dark brown, "
            "black, blue-gray (indicating dermal melanin / melanophages), red (erythema / vascularity), or white (scar-like regression)—is "
            "strongly suspicious for melanoma. High standard deviations in HSV channels correlate directly with multi-chromatic lesion architecture."
        ),
    },
    {
        "id": "KB-DIA-006",
        "topic": "diameter_calibration",
        "feature": "D",
        "source_type": "consensus_guideline",
        "year": 2024,
        "reliability": 0.95,
        "title": "Diameter Assessment and Calibration Requirements",
        "text": (
            "The traditional ABCD threshold cites lesion diameter exceeding 6 mm (the approximate diameter of a pencil eraser) as a "
            "warning sign. However, early 'small-diameter melanomas' (<6 mm) increasingly occur. Crucially, physical millimeter dimensions "
            "cannot be fabricated or inferred from raw pixel counts without a verified physical calibration marker or graduated millimeter ruler. "
            "Uncalibrated images can only report pixel extents."
        ),
    },
    {
        "id": "KB-EVO-007",
        "topic": "evolution_tracking",
        "feature": "E",
        "source_type": "clinical_guideline",
        "year": 2024,
        "reliability": 0.99,
        "title": "Evolution and Longitudinal Dermatoscopic Monitoring",
        "text": (
            "Evolution (E)—documented changes in size, shape, surface contour, or color over time—is the single most sensitive indicator "
            "of early cutaneous melanoma. Assessing evolution requires strict temporal comparison between registered images taken at different "
            "visits. A single static image cannot evaluate evolution; longitudinal image pairs are mandatory to establish whether significant change occurred."
        ),
    },
    {
        "id": "KB-LIMIT-008",
        "topic": "model_limitations",
        "feature": "safety",
        "source_type": "safety_standard",
        "year": 2025,
        "reliability": 0.99,
        "title": "AI Clinical Limitations and Diagnostic Boundaries",
        "text": (
            "AI-assisted skin lesion assessment models are decision-support research tools, not autonomous medical diagnostic devices. "
            "Performance can be compromised by artifacts such as thick hair, air bubbles, optical vignetting, gel reflections, or poor focus. "
            "Rare melanoma variants (e.g. amelanotic melanoma, desmoplastic melanoma) may present without typical pigment features. "
            "All concerning findings mandate formal in-person examination by a qualified dermatologist."
        ),
    },
]
