"""
Agentic RAG Module for CEFM Melanoma Framework.

Provides autonomous, bounded retrieval with evidence evaluation,
provenance tracking, and research trace generation.
"""

from app.rag.agentic.clinical_agent import (
    ClinicalRAGAgent,
    MAX_RETRIEVAL_ROUNDS,
    RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
)
from app.rag.agentic.evaluator import (
    EvidenceEvaluator,
    DEFAULT_RELEVANCE_THRESHOLD,
)

__all__ = [
    "ClinicalRAGAgent",
    "EvidenceEvaluator",
    "MAX_RETRIEVAL_ROUNDS",
    "RELEVANCE_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_RELEVANCE_THRESHOLD",
]
