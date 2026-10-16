from __future__ import annotations

import hashlib
import json
import re

import pytest

from scripts import run_rag_benchmark as benchmark


BENCHMARK_PATH = (
    benchmark.BASE_DIR / "tests" / "data" / "agentic_rag_benchmark.json"
)


def _single_query_benchmark(tmp_path):
    corpus = [
        {
            "_id": "0123456789abcdef01234567",
            "source": "guide.pdf",
            "chunk_index": 0,
            "text": "Melanoma guidance.",
            "embedding": [1.0, 0.0],
        }
    ]
    document_id = corpus[0]["_id"]
    data = {
        "benchmark_id": "unit-test",
        "id_scheme": "MongoDB ObjectId hex string",
        "expected_corpus_sha256": benchmark.corpus_fingerprint(corpus),
        "corpus_snapshot": [
            {
                "document_id": document_id,
                "source": "guide.pdf",
                "chunk_index": 0,
            }
        ],
        "queries": [
            {
                "query_id": "Q001",
                "query": "melanoma guidance",
                "category": "classification",
                "relevant_documents": [document_id],
                "relevance_grades": {document_id: 3},
                "agentic_input": {
                    "classification": {"predicted_class": "mel"},
                    "abcde": {},
                    "evolution": {"available": False},
                },
            }
        ],
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, corpus, data


def test_fixed_benchmark_json_schema_ids_and_load_are_deterministic():
    raw = BENCHMARK_PATH.read_bytes()
    data = json.loads(raw)

    valid_ids = benchmark.validate_benchmark_schema(data)

    assert len(data["queries"]) == 15
    assert len(valid_ids) == len(data["corpus_snapshot"]) == 14
    assert all(re.fullmatch(r"[0-9a-f]{24}", identifier) for identifier in valid_ids)
    assert len({item["query_id"] for item in data["queries"]}) == len(data["queries"])
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(
        BENCHMARK_PATH.read_bytes()
    ).hexdigest()


def test_benchmark_schema_rejects_duplicate_query_ids():
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    data["queries"][1]["query_id"] = data["queries"][0]["query_id"]

    with pytest.raises(benchmark.BenchmarkValidationError, match="Duplicate benchmark query ID"):
        benchmark.validate_benchmark_schema(data)


def test_benchmark_schema_rejects_nonexistent_relevant_document():
    data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    data["queries"][0]["relevant_documents"] = ["000000000000000000000000"]
    data["queries"][0]["relevance_grades"] = {
        "000000000000000000000000": 3
    }

    with pytest.raises(benchmark.BenchmarkValidationError, match="nonexistent corpus IDs"):
        benchmark.validate_benchmark_schema(data)


def test_precision_recall_mrr_and_ndcg_formulas():
    relevant = {"doc-a", "doc-b"}
    grades = {"doc-a": 3, "doc-b": 1}
    ranking = ["doc-c", "doc-b", "doc-a"]

    assert benchmark.precision_at_k(ranking, relevant, 1) == 0.0
    assert benchmark.precision_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert benchmark.recall_at_k(ranking, relevant, 1) == 0.0
    assert benchmark.recall_at_k(ranking, relevant, 3) == 1.0
    assert benchmark.mean_reciprocal_rank(ranking, relevant) == 0.5
    assert 0.0 < benchmark.ndcg_at_k(ranking, grades, 3) < 1.0
    assert benchmark.ndcg_at_k(["doc-a", "doc-b"], grades, 3) == 1.0


def test_second_round_recovery_rate_and_empty_denominator():
    rows = [
        {
            "round1_sufficient": False,
            "final_sufficient": True,
        },
        {
            "round1_sufficient": False,
            "final_sufficient": False,
        },
        {
            "round1_sufficient": True,
            "final_sufficient": True,
        },
    ]

    assert benchmark.second_round_recovery_rate(rows) == {
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
    }
    assert benchmark.second_round_recovery_rate(
        [{"round1_sufficient": True, "final_sufficient": True}]
    ) == {"value": None, "numerator": 0, "denominator": 0}


def test_both_systems_receive_same_fixed_query_and_corpus(tmp_path):
    dataset_path, corpus, data = _single_query_benchmark(tmp_path)
    baseline_calls = []
    agent_calls = []
    result_document = {
        "source": "guide.pdf",
        "chunk_index": 0,
        "score": 0.91,
    }

    def fake_retriever(query, top_k):
        baseline_calls.append((query, top_k))
        return [result_document]

    class FakeAgent:
        def __init__(self, **kwargs):
            assert kwargs == {
                "max_rounds": benchmark.MAX_RETRIEVAL_ROUNDS,
                "relevance_threshold": benchmark.RELEVANCE_THRESHOLD,
                "top_k": benchmark.TOP_K,
            }

        def run(self, classification, abcde, evolution, top_k):
            agent_calls.append((classification, abcde, evolution, top_k))
            return {
                "agent_trace": {
                    "rounds": [
                        {
                            "iteration": 1,
                            "queries": ["targeted melanoma guidance"],
                            "retrieved_documents": [result_document],
                            "similarity_scores": [0.91],
                            "missing_domains": [],
                            "decision": "stop",
                        }
                    ]
                },
                "provenance": [{"category": "[RETRIEVED]", "source": "guide.pdf"}],
                "rounds_executed": 1,
                "sufficiency": "sufficient",
                "documents": [result_document],
            }

    result = benchmark.run_benchmark(
        dataset_path,
        tmp_path / "out",
        corpus_documents=corpus,
        conventional_retriever=fake_retriever,
        agent_factory=FakeAgent,
        write_outputs=False,
    )

    query = data["queries"][0]
    assert baseline_calls == [(query["query"], benchmark.TOP_K)]
    assert agent_calls == [
        (
            query["agentic_input"]["classification"],
            query["agentic_input"]["abcde"],
            query["agentic_input"]["evolution"],
            benchmark.TOP_K,
        )
    ]
    assert result["per_query_results"][0]["conventional_recall_at_1"] == 1.0
    assert result["per_query_results"][0]["agentic_recall_at_1"] == 1.0
