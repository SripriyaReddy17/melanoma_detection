"""
Evidence Evaluator for Agentic RAG.

Inspects retrieved medical documents and similarity scores to determine
evidence relevance, sufficiency, and domain coverage.

NOTE: RELEVANCE_THRESHOLD is an engineering configuration used for
retrieval filtering and sufficiency heuristics; it is NOT a clinically
validated diagnostic threshold.
"""

from typing import List, Dict, Any, Optional

# Engineering configuration for vector similarity cutoff.
# Configurable and not to be described as a clinical diagnostic standard.
DEFAULT_RELEVANCE_THRESHOLD: float = 0.45

# Domain keyword lexicons for relevance mapping
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "classification": [
        "melanoma", "malignant", "nevus", "nevi", "benign",
        "melanocytic", "lesion", "excision", "radial", "vertical",
        "metastatic", "histology", "dermatology", "triage",
    ],
    "asymmetry": [
        "asymmetry", "asymmetric", "symmetry", "orthogonal", "axis",
        "axes", "mirror", "contour", "geometric", "imbalance",
        "half", "halves", "differential", "pigment distribution",
    ],
    "border": [
        "border", "notch", "notched", "scalloped", "contour",
        "margin", "margins", "roughness", "boundary", "irregularity",
        "curvature", "abrupt", "cutoff",
    ],
    "color": [
        "color", "variegation", "chromatic", "pigment", "melanin",
        "hsv", "diversity", "shades", "erythema", "blue-gray",
        "hypopigmentation", "heterogeneity",
    ],
    "diameter": [
        "diameter", "6 mm", "6mm", "millimeter", "measurement",
        "calibration", "extent", "size", "ruler", "pixels",
    ],
    "evolution": [
        "evolution", "temporal", "longitudinal", "sequential",
        "change", "monitoring", "stability", "prior", "baseline",
    ],
    "safety": [
        "limitation", "limitations", "artifact", "dermatologist",
        "decision-support", "biopsy", "disclaimer", "histopathology",
    ],
}


class EvidenceEvaluator:
    """
    Evaluates retrieved clinical literature documents against target domains.
    Determines whether evidence is sufficient or if an additional retrieval
    round is required.
    """

    def __init__(self, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD):
        """
        Initialize the evaluator with an engineering relevance threshold.

        Args:
            relevance_threshold: Cosine similarity cutoff for considering
                a retrieved document clinically relevant.
        """
        self.relevance_threshold = float(relevance_threshold)

    def is_document_relevant(self, document: Dict[str, Any], domain: str) -> bool:
        """
        Determine whether a single document is relevant to a requested domain.

        Requires:
        1. Similarity score >= relevance_threshold
        2. Presence of domain-relevant clinical terminology in document text
        """
        if not document:
            return False

        score = float(document.get("score", 0.0) or 0.0)
        if score < self.relevance_threshold:
            return False

        text = str(document.get("text", "")).lower()
        if not text:
            return False

        keywords = DOMAIN_KEYWORDS.get(domain.lower(), [])
        if not keywords:
            # If domain has no specific keyword list, accept score threshold
            return True

        return any(kw in text for kw in keywords)

    def evaluate_evidence(
        self,
        retrieved_documents: List[Dict[str, Any]],
        target_domains: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a collection of retrieved documents against target domains.

        Args:
            retrieved_documents: List of document dicts with 'text', 'score', 'source'
            target_domains: List of domain keys required for this case
                (e.g., ['classification', 'asymmetry'])

        Returns:
            Dict containing:
                is_sufficient (bool): True if all target domains have relevant evidence
                covered_domains (List[str]): Domains with at least one relevant document
                missing_domains (List[str]): Domains lacking sufficient relevant evidence
                highest_score (float): Highest similarity score among documents
                domain_matches (Dict[str, List[Dict[str, Any]]]): Matching docs per domain
                similarity_scores (List[float]): All scores in retrieved documents
        """
        if not retrieved_documents:
            return {
                "is_sufficient": False,
                "covered_domains": [],
                "missing_domains": list(target_domains),
                "highest_score": 0.0,
                "domain_matches": {d: [] for d in target_domains},
                "similarity_scores": [],
            }

        scores = [
            float(doc.get("score", 0.0) or 0.0)
            for doc in retrieved_documents
            if doc
        ]
        highest_score = max(scores) if scores else 0.0

        domain_matches: Dict[str, List[Dict[str, Any]]] = {
            domain: [] for domain in target_domains
        }

        covered_domains: List[str] = []
        missing_domains: List[str] = []

        for domain in target_domains:
            matches = [
                doc
                for doc in retrieved_documents
                if self.is_document_relevant(doc, domain)
            ]
            domain_matches[domain] = matches
            if matches:
                covered_domains.append(domain)
            else:
                missing_domains.append(domain)

        # Evidence is sufficient if all requested target domains are covered
        # and at least one document meets the relevance threshold.
        is_sufficient = (
            len(missing_domains) == 0
            and highest_score >= self.relevance_threshold
            and len(target_domains) > 0
        )

        return {
            "is_sufficient": is_sufficient,
            "covered_domains": covered_domains,
            "missing_domains": missing_domains,
            "highest_score": round(highest_score, 4),
            "domain_matches": domain_matches,
            "similarity_scores": scores,
        }

    def should_retrieve_again(
        self,
        evaluation: Dict[str, Any],
        current_round: int,
        max_rounds: int = 2,
    ) -> bool:
        """
        Determine whether another retrieval round should be triggered.

        Args:
            evaluation: Result from evaluate_evidence()
            current_round: 1-indexed current retrieval round
            max_rounds: Maximum allowed rounds (strictly bounded)

        Returns:
            bool: True if missing evidence warrants another round and round < max_rounds.
        """
        if current_round >= max_rounds:
            return False
        return not evaluation.get("is_sufficient", False)
