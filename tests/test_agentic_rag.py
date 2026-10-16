"""
Unit tests for Agentic RAG implementation (CEFM Melanoma Framework).

Verifies ClinicalRAGAgent, EvidenceEvaluator, bounded retrieval loop,
provenance attribution, trace structure, and medical safety guardrails.
Does NOT require an active MongoDB connection.
"""

import copy
import pytest
from unittest.mock import MagicMock

from app.rag.agentic.clinical_agent import (
    ClinicalRAGAgent,
    MAX_RETRIEVAL_ROUNDS,
    RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
)
from app.rag.agentic.evaluator import EvidenceEvaluator


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_classification():
    return {
        "predicted_class": "mel",
        "melanoma_probability": 0.85,
        "risk_level": "high",
    }


@pytest.fixture
def sample_abcde_anomalous():
    return {
        "asymmetry": 0.72,
        "border_irregularity": 0.65,
        "color_index": 0.55,
        "diameter_mm": 8.2,
        "diameter_pixels": 420.0,
        "diameter_mm_available": True,
        "calibration_used": True,
    }


@pytest.fixture
def sample_evolution_available():
    return {
        "available": True,
        "change_detected": True,
        "overall_change": 0.35,
    }


@pytest.fixture
def sample_evolution_unavailable():
    return {
        "available": False,
        "reason": "Previous image not provided",
    }


# ============================================================
# 15 TEST CASES
# ============================================================

def test_1_agent_initializes():
    """1. Agent initializes with default bounded configuration."""
    agent = ClinicalRAGAgent()
    assert agent.max_rounds == MAX_RETRIEVAL_ROUNDS
    assert agent.max_rounds == 2
    assert agent.relevance_threshold == RELEVANCE_THRESHOLD
    assert agent.relevance_threshold == 0.45
    assert agent.top_k == DEFAULT_TOP_K
    assert agent.top_k == 3
    assert isinstance(agent.evaluator, EvidenceEvaluator)
    assert agent.evaluator.relevance_threshold == 0.45


def test_agent_rejects_retrieval_round_limits_above_two():
    with pytest.raises(ValueError, match="between 1 and 2"):
        ClinicalRAGAgent(max_rounds=3)


def test_2_targeted_queries_are_generated(sample_classification, sample_abcde_anomalous, sample_evolution_available):
    """2. Targeted clinical queries are generated per domain rather than one monolith."""
    agent = ClinicalRAGAgent()
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution=sample_evolution_available,
    )
    assert isinstance(queries, list)
    assert len(queries) >= 3
    domains = [q["domain"] for q in queries]
    assert "classification" in domains
    for q in queries:
        assert "domain" in q
        assert "query" in q
        assert len(q["query"]) > 10


def test_3_asymmetry_query_generation(sample_classification):
    """3. Asymmetry query generation occurs for anomalous asymmetry score."""
    agent = ClinicalRAGAgent()
    abcde = {"asymmetry": 0.70}
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=abcde,
        evolution={},
    )
    asym_queries = [q for q in queries if q["domain"] == "asymmetry"]
    assert len(asym_queries) == 1
    query_text = asym_queries[0]["query"].lower()
    assert "asymmetry" in query_text
    assert "orthogonal" in query_text


def test_4_border_query_generation(sample_classification):
    """4. Border query generation occurs for anomalous border irregularity."""
    agent = ClinicalRAGAgent()
    abcde = {"border_irregularity": 0.60}
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=abcde,
        evolution={},
    )
    border_queries = [q for q in queries if q["domain"] == "border"]
    assert len(border_queries) == 1
    query_text = border_queries[0]["query"].lower()
    assert "border" in query_text
    assert "notched" in query_text or "irregular" in query_text


def test_5_color_query_generation(sample_classification):
    """5. Color query generation occurs for anomalous color index."""
    agent = ClinicalRAGAgent()
    abcde = {"color_index": 0.50}
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=abcde,
        evolution={},
    )
    color_queries = [q for q in queries if q["domain"] == "color"]
    assert len(color_queries) == 1
    query_text = color_queries[0]["query"].lower()
    assert "color" in query_text
    assert "variegation" in query_text or "pigment" in query_text


def test_6_diameter_query_generation(sample_classification):
    """6. Diameter query generation occurs only with valid physical calibration."""
    agent = ClinicalRAGAgent()
    abcde = {
        "diameter": {
            "max_diameter_mm": 7.5,
            "calibration_used": True,
            "available": True,
        }
    }
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=abcde,
        evolution={},
    )
    dia_queries = [q for q in queries if q["domain"] == "diameter"]
    assert len(dia_queries) == 1
    query_text = dia_queries[0]["query"].lower()
    assert "diameter" in query_text
    assert "6mm" in query_text or "calibration" in query_text


def test_7_evolution_is_not_fabricated(sample_classification, sample_abcde_anomalous, sample_evolution_unavailable):
    """7. Evolution query is NOT generated and not fabricated when unavailable."""
    agent = ClinicalRAGAgent()
    queries = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution=sample_evolution_unavailable,
    )
    domains = [q["domain"] for q in queries]
    assert "evolution" not in domains

    # Also test empty evolution dict
    queries_empty = agent.plan_initial_queries(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution={},
    )
    assert "evolution" not in [q["domain"] for q in queries_empty]


def test_8_existing_retriever_is_called(sample_classification, sample_abcde_anomalous):
    """8. The existing retriever function is called with planned queries."""
    mock_retriever = MagicMock(return_value=[
        {"source": "AAD_ABCDE_Melanoma.pdf", "chunk_index": 1, "text": "Melanoma clinical guidelines.", "score": 0.60}
    ])
    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution={"available": False},
        top_k=3,
    )
    assert mock_retriever.called
    assert mock_retriever.call_count >= 1
    # Check that query arguments were passed
    first_call_args = mock_retriever.call_args_list[0]
    assert isinstance(first_call_args[0][0], str)
    assert first_call_args[0][1] == 3


def test_9_sufficient_evidence_stops_retrieval(sample_classification):
    """9. Sufficient evidence in Round 1 stops retrieval without triggering Round 2."""
    def mock_retriever(query, top_k):
        # Always return highly relevant matching documents
        return [
            {
                "source": "AAD_ABCDE_Melanoma.pdf",
                "chunk_index": 2,
                "text": "Malignant melanoma clinical presentation and surgical excision guidelines. Lesion asymmetry along orthogonal axes.",
                "score": 0.72,
            }
        ]

    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde={"asymmetry": 0.60},
        evolution={"available": False},
    )

    assert result["rounds_executed"] == 1
    trace = result["agent_trace"]
    assert len(trace["rounds"]) == 1
    assert trace["rounds"][0]["decision"] == "stop"
    assert result["sufficiency"] == "sufficient"


def test_10_insufficient_evidence_triggers_second_round(sample_classification):
    """10. Insufficient evidence in Round 1 triggers a focused Round 2."""
    call_counts = {"count": 0}

    def mock_retriever(query, top_k):
        call_counts["count"] += 1
        if call_counts["count"] <= 2:
            # Round 1 returns low similarity / off-target documents
            return [{
                "source": "general.pdf",
                "chunk_index": 0,
                "text": "General skin hydration and anatomy notes.",
                "score": 0.25,  # Below RELEVANCE_THRESHOLD 0.45
            }]
        else:
            # Round 2 returns relevant documents
            return [{
                "source": "AAD_ABCDE_Melanoma.pdf",
                "chunk_index": 3,
                "text": "Cutaneous malignant melanoma clinical presentation and asymmetry orthogonal axes.",
                "score": 0.68,
            }]

    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde={"asymmetry": 0.60},
        evolution={"available": False},
    )

    assert result["rounds_executed"] == 2
    trace = result["agent_trace"]
    assert len(trace["rounds"]) == 2
    assert trace["rounds"][0]["decision"] == "retrieve_again"
    assert trace["rounds"][1]["decision"] == "stop"


def test_11_maximum_two_rounds_enforced(sample_classification):
    """11. Maximum 2 rounds strictly enforced even if evidence remains insufficient."""
    def always_empty_retriever(query, top_k):
        return []

    agent = ClinicalRAGAgent(max_rounds=2, retriever_fn=always_empty_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde={"asymmetry": 0.60},
        evolution={"available": False},
    )

    assert result["rounds_executed"] == 2
    assert len(result["agent_trace"]["rounds"]) == 2
    assert result["agent_trace"]["rounds"][1]["decision"] == "stop"
    assert result["sufficiency"] == "insufficient"


def test_12_agent_trace_is_generated(sample_classification, sample_abcde_anomalous):
    """12. Agent trace has required schema and does NOT expose chain-of-thought."""
    mock_retriever = MagicMock(return_value=[
        {"source": "AAD_ABCDE_Melanoma.pdf", "chunk_index": 1, "text": "Melanoma excision guidelines.", "score": 0.65}
    ])
    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution={"available": False},
    )

    trace = result["agent_trace"]
    assert trace["agent_type"] == "clinical_rag_agent"
    assert trace["max_rounds"] == 2
    assert isinstance(trace["rounds"], list)
    assert len(trace["rounds"]) >= 1

    for r in trace["rounds"]:
        assert "iteration" in r
        assert "queries" in r
        assert "retrieved_documents" in r
        assert "similarity_scores" in r
        assert "missing_domains" in r
        assert "decision" in r
        assert r["decision"] in ["retrieve_again", "stop"]
        # Ensure no raw chain-of-thought text dumps
        assert "thought" not in r
        assert "scratchpad" not in r


def test_13_model_prediction_remains_unchanged(sample_classification, sample_abcde_anomalous):
    """13. Agent does NOT modify classification prediction, probability, or ABCDE numbers."""
    orig_classification = copy.deepcopy(sample_classification)
    orig_abcde = copy.deepcopy(sample_abcde_anomalous)

    mock_retriever = MagicMock(return_value=[
        {"source": "AAD_ABCDE_Melanoma.pdf", "chunk_index": 1, "text": "Melanoma guidelines.", "score": 0.60}
    ])
    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution={"available": False},
    )

    # Strictly assert identical contents
    assert sample_classification == orig_classification
    assert sample_abcde_anomalous == orig_abcde
    assert sample_classification["predicted_class"] == "mel"
    assert sample_classification["melanoma_probability"] == 0.85


def test_14_unavailable_information_remains_unavailable(sample_classification):
    """14. Uncalibrated diameter and unavailable evolution remain strictly categorized as [UNAVAILABLE]."""
    uncalibrated_abcde = {
        "asymmetry": 0.40,
        "diameter_pixels": 320.0,
        "calibration_used": False,
        "diameter_mm_available": False,
    }
    unavailable_evolution = {
        "available": False,
    }

    mock_retriever = MagicMock(return_value=[
        {"source": "AAD_ABCDE_Melanoma.pdf", "chunk_index": 1, "text": "Melanoma guidelines.", "score": 0.55}
    ])
    agent = ClinicalRAGAgent(retriever_fn=mock_retriever)
    result = agent.run(
        classification=sample_classification,
        abcde=uncalibrated_abcde,
        evolution=unavailable_evolution,
    )

    provenance = result["provenance"]
    categories = {p["feature"]: p["category"] for p in provenance}
    statements = {p["feature"]: p["statement"] for p in provenance}

    assert categories.get("evolution") == "[UNAVAILABLE]"
    assert "single image" in statements.get("evolution", "").lower()

    assert categories.get("diameter_mm") == "[UNAVAILABLE]"
    assert "calibration" in statements.get("diameter_mm", "").lower()


def test_15_empty_retrieval_handled_safely(sample_classification, sample_abcde_anomalous):
    """15. Empty database retrieval returns safely without crashing and produces valid trace."""
    empty_retriever = MagicMock(return_value=[])
    agent = ClinicalRAGAgent(retriever_fn=empty_retriever)

    result = agent.run(
        classification=sample_classification,
        abcde=sample_abcde_anomalous,
        evolution={"available": False},
    )

    assert isinstance(result, dict)
    assert result["documents"] == []
    assert result["sufficiency"] == "insufficient"
    assert "agent_trace" in result
    assert result["rounds_executed"] == 2
    assert result["agent_trace"]["rounds"][-1]["decision"] == "stop"
