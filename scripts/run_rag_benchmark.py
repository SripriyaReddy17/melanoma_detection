"""Run the fixed-corpus conventional versus Agentic RAG retrieval benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from app.rag.agentic.clinical_agent import (
    ClinicalRAGAgent,
    MAX_RETRIEVAL_ROUNDS,
    RELEVANCE_THRESHOLD,
)
from app.rag.embeddings import MODEL_NAME
from app.rag.retrieval import retrieve_documents


DEFAULT_DATASET = BASE_DIR / "tests" / "data" / "agentic_rag_benchmark.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "rag"
TOP_K = 5
RANDOM_SEED = 42
METRIC_CUTOFFS = (1, 3, 5)
AGENT_LOGGER = logging.getLogger("app.rag.agentic.clinical_agent")


class BenchmarkValidationError(ValueError):
    """Raised when the fixed benchmark or current corpus fails validation."""


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


def _document_id(document: dict[str, Any]) -> str:
    value = document.get("_id", document.get("document_id"))
    if value is None:
        raise BenchmarkValidationError("Corpus document is missing its MongoDB _id.")
    return str(value)


def _natural_key(document: dict[str, Any]) -> tuple[str, str]:
    source = document.get("source")
    chunk_index = document.get("chunk_index")
    if source is None or chunk_index is None:
        raise BenchmarkValidationError(
            f"Corpus document {_document_id(document)} lacks source/chunk_index."
        )
    return str(source), str(chunk_index)


def corpus_fingerprint(documents: list[dict[str, Any]]) -> str:
    """Fingerprint current corpus identity, text, and stored vectors."""
    records = [
        {
            "document_id": _document_id(document),
            "source": document.get("source"),
            "chunk_index": document.get("chunk_index"),
            "text": document.get("text"),
            "embedding": document.get("embedding"),
        }
        for document in sorted(documents, key=_document_id)
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_benchmark_schema(benchmark: dict[str, Any]) -> set[str]:
    """Validate fixed query/label structure using the declared corpus snapshot."""
    if not isinstance(benchmark, dict):
        raise BenchmarkValidationError("Benchmark root must be a JSON object.")
    snapshot = benchmark.get("corpus_snapshot")
    queries = benchmark.get("queries")
    if not isinstance(snapshot, list) or not snapshot:
        raise BenchmarkValidationError("Benchmark corpus_snapshot must be non-empty.")
    if not isinstance(queries, list) or not queries:
        raise BenchmarkValidationError("Benchmark queries must be non-empty.")

    document_ids: set[str] = set()
    for document in snapshot:
        if not isinstance(document, dict):
            raise BenchmarkValidationError("Each corpus snapshot record must be an object.")
        identifier = str(document.get("document_id", ""))
        if not re.fullmatch(r"[0-9a-f]{24}", identifier):
            raise BenchmarkValidationError(
                f"Invalid MongoDB ObjectId hex string in corpus_snapshot: {identifier!r}"
            )
        if identifier in document_ids:
            raise BenchmarkValidationError(
                f"Duplicate corpus document ID in snapshot: {identifier}"
            )
        if not document.get("source") or document.get("chunk_index") is None:
            raise BenchmarkValidationError(
                f"Corpus snapshot document {identifier} needs source and chunk_index."
            )
        document_ids.add(identifier)

    query_ids: set[str] = set()
    for item in queries:
        if not isinstance(item, dict):
            raise BenchmarkValidationError("Each benchmark query must be an object.")
        query_id = item.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise BenchmarkValidationError("Every query needs a non-empty query_id.")
        if query_id in query_ids:
            raise BenchmarkValidationError(f"Duplicate benchmark query ID: {query_id}")
        query_ids.add(query_id)
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise BenchmarkValidationError(f"{query_id} has an empty query.")
        if not isinstance(item.get("category"), str) or not item["category"].strip():
            raise BenchmarkValidationError(f"{query_id} has an empty category.")
        relevant = item.get("relevant_documents")
        grades = item.get("relevance_grades")
        if not isinstance(relevant, list) or not relevant:
            raise BenchmarkValidationError(
                f"{query_id} must have manually labeled relevant_documents."
            )
        relevant_ids = [str(identifier) for identifier in relevant]
        if len(set(relevant_ids)) != len(relevant_ids):
            raise BenchmarkValidationError(f"{query_id} repeats a relevant document.")
        unknown_ids = set(relevant_ids) - document_ids
        if unknown_ids:
            raise BenchmarkValidationError(
                f"{query_id} references nonexistent corpus IDs: {sorted(unknown_ids)}"
            )
        if not isinstance(grades, dict) or set(grades) != set(relevant_ids):
            raise BenchmarkValidationError(
                f"{query_id} relevance_grades keys must exactly match "
                "relevant_documents."
            )
        for identifier, grade in grades.items():
            if (
                isinstance(grade, bool)
                or not isinstance(grade, int)
                or not 1 <= grade <= 3
            ):
                raise BenchmarkValidationError(
                    f"{query_id} has invalid relevance grade for {identifier}; "
                    "grades must be integers 1, 2, or 3."
                )
        agentic_input = item.get("agentic_input")
        if not isinstance(agentic_input, dict) or any(
            not isinstance(agentic_input.get(key), dict)
            for key in ("classification", "abcde", "evolution")
        ):
            raise BenchmarkValidationError(
                f"{query_id} must define structured classification, abcde, and "
                "evolution inputs for the actual ClinicalRAGAgent."
            )
    return document_ids


def validate_benchmark(
    benchmark: dict[str, Any],
    corpus_documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate query labels against actual corpus IDs before retrieval."""
    validate_benchmark_schema(benchmark)
    if not isinstance(benchmark, dict):
        raise BenchmarkValidationError("Benchmark root must be a JSON object.")

    expected_snapshot = benchmark.get("corpus_snapshot")
    queries = benchmark.get("queries")
    if not isinstance(expected_snapshot, list) or not expected_snapshot:
        raise BenchmarkValidationError("Benchmark corpus_snapshot must be non-empty.")
    if not isinstance(queries, list) or not queries:
        raise BenchmarkValidationError("Benchmark queries must be non-empty.")

    actual_by_id: dict[str, dict[str, Any]] = {}
    by_natural_key: dict[tuple[str, str], dict[str, Any]] = {}
    for document in corpus_documents:
        identifier = _document_id(document)
        natural_key = _natural_key(document)
        if identifier in actual_by_id:
            raise BenchmarkValidationError(f"Duplicate corpus document ID: {identifier}")
        if natural_key in by_natural_key:
            raise BenchmarkValidationError(
                f"Duplicate corpus source/chunk key: {natural_key}"
            )
        if not isinstance(document.get("text"), str) or not document["text"].strip():
            raise BenchmarkValidationError(
                f"Corpus document {identifier} has no text."
            )
        actual_by_id[identifier] = document
        by_natural_key[natural_key] = document

    expected_records = {
        str(item.get("document_id")): (
            str(item.get("source")),
            str(item.get("chunk_index")),
        )
        for item in expected_snapshot
    }
    actual_records = {
        identifier: _natural_key(document)
        for identifier, document in actual_by_id.items()
    }
    if len(expected_records) != len(expected_snapshot):
        raise BenchmarkValidationError("Benchmark corpus_snapshot has duplicate IDs.")
    if expected_records != actual_records:
        raise BenchmarkValidationError(
            "Current MongoDB document IDs/source/chunk indexes differ from the "
            "fixed benchmark corpus_snapshot."
        )

    actual_fingerprint = corpus_fingerprint(corpus_documents)
    expected_fingerprint = benchmark.get("expected_corpus_sha256")
    if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
        raise BenchmarkValidationError(
            "Current MongoDB corpus fingerprint differs from the fixed benchmark "
            f"(expected {expected_fingerprint!r}, got {actual_fingerprint})."
        )

    query_ids: set[str] = set()
    for item in queries:
        if not isinstance(item, dict):
            raise BenchmarkValidationError("Each benchmark query must be an object.")
        query_id = item.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise BenchmarkValidationError("Every query needs a non-empty query_id.")
        if query_id in query_ids:
            raise BenchmarkValidationError(f"Duplicate benchmark query ID: {query_id}")
        query_ids.add(query_id)
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise BenchmarkValidationError(f"{query_id} has an empty query.")
        if not isinstance(item.get("category"), str) or not item["category"].strip():
            raise BenchmarkValidationError(f"{query_id} has an empty category.")

        relevant = item.get("relevant_documents")
        grades = item.get("relevance_grades")
        if not isinstance(relevant, list) or not relevant:
            raise BenchmarkValidationError(
                f"{query_id} must have manually labeled relevant_documents."
            )
        relevant_ids = [str(identifier) for identifier in relevant]
        if len(set(relevant_ids)) != len(relevant_ids):
            raise BenchmarkValidationError(f"{query_id} repeats a relevant document.")
        unknown_ids = set(relevant_ids) - actual_by_id.keys()
        if unknown_ids:
            raise BenchmarkValidationError(
                f"{query_id} references nonexistent corpus IDs: {sorted(unknown_ids)}"
            )
        if not isinstance(grades, dict) or set(grades) != set(relevant_ids):
            raise BenchmarkValidationError(
                f"{query_id} relevance_grades keys must exactly match "
                "relevant_documents."
            )
        for identifier, grade in grades.items():
            if isinstance(grade, bool) or not isinstance(grade, int) or not 1 <= grade <= 3:
                raise BenchmarkValidationError(
                    f"{query_id} has invalid relevance grade for {identifier}; "
                    "grades must be integers 1, 2, or 3."
                )

        agentic_input = item.get("agentic_input")
        if not isinstance(agentic_input, dict) or any(
            not isinstance(agentic_input.get(key), dict)
            for key in ("classification", "abcde", "evolution")
        ):
            raise BenchmarkValidationError(
                f"{query_id} must define structured classification, abcde, and "
                "evolution inputs for the actual ClinicalRAGAgent."
            )

    return by_natural_key


def load_benchmark(
    path: Path,
    corpus_documents: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, dict[tuple[str, str], dict[str, Any]]]:
    raw = path.read_bytes()
    try:
        benchmark = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkValidationError(f"Invalid benchmark JSON: {exc}") from exc
    lookup = validate_benchmark(benchmark, corpus_documents)
    return benchmark, hashlib.sha256(raw).hexdigest(), lookup


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(identifier in relevant_ids for identifier in retrieved_ids[:k]) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant_ids:
        return 0.0
    return sum(identifier in relevant_ids for identifier in retrieved_ids[:k]) / len(
        relevant_ids
    )


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    def dcg(grades: list[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    actual = [relevance_grades.get(identifier, 0) for identifier in retrieved_ids[:k]]
    ideal = sorted(relevance_grades.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    relevance_grades: dict[str, int],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in METRIC_CUTOFFS:
        result[f"precision_at_{cutoff}"] = precision_at_k(
            retrieved_ids, relevant_ids, cutoff
        )
        result[f"recall_at_{cutoff}"] = recall_at_k(
            retrieved_ids, relevant_ids, cutoff
        )
        if cutoff in (3, 5):
            result[f"ndcg_at_{cutoff}"] = ndcg_at_k(
                retrieved_ids, relevance_grades, cutoff
            )
    result["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    metric_names = [
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "ndcg_at_3",
        "ndcg_at_5",
    ]
    return {
        name: _mean([float(row[f"{prefix}_{name}"]) for row in rows])
        for name in metric_names
    }


def second_round_recovery_rate(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Return recovery among queries with an observed insufficient first round."""
    first_round_insufficient = [
        row
        for row in rows
        if row.get("round1_sufficient") is False
    ]
    recovered = sum(bool(row.get("final_sufficient")) for row in first_round_insufficient)
    return {
        "value": (
            recovered / len(first_round_insufficient)
            if first_round_insufficient
            else None
        ),
        "numerator": recovered,
        "denominator": len(first_round_insufficient),
    }


def _result_identity(
    item: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = _natural_key(item)
    if key not in lookup:
        raise BenchmarkValidationError(
            f"Retriever returned a document outside the corpus snapshot: {key}"
        )
    corpus_document = lookup[key]
    return {
        "document_id": _document_id(corpus_document),
        "source": key[0],
        "chunk_index": corpus_document["chunk_index"],
        "score": float(item.get("score", 0.0) or 0.0),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _metric_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_{name}"
        for name in (
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "ndcg_at_3",
            "ndcg_at_5",
        )
    ]


def _write_outputs(
    output_dir: Path,
    results: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agentic_rag_benchmark_results.json"
    csv_path = output_dir / "agentic_rag_benchmark_results.csv"
    summary_path = output_dir / "agentic_rag_benchmark_summary.md"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    base_columns = [
        "query_id",
        "category",
        "query",
        *_metric_columns("conventional"),
        *_metric_columns("agentic"),
        "rounds_used",
        "second_round_triggered",
        "round1_sufficient",
        "final_sufficient",
        "conventional_error",
        "agentic_error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=base_columns)
        writer.writeheader()
        for row in results["per_query_results"]:
            writer.writerow({column: row.get(column) for column in base_columns})

    summary = results["summary"]
    conventional = summary["conventional"]
    agentic = summary["agentic"]
    lines = [
        "# Conventional vs. Agentic RAG Retrieval Benchmark",
        "",
        f"- Benchmark queries: {results['benchmark_metadata']['query_count']}",
        f"- Corpus documents: {results['corpus_metadata']['document_count']}",
        f"- Corpus sources: {', '.join(results['corpus_metadata']['sources'])}",
        f"- Corpus SHA-256: `{results['corpus_metadata']['sha256']}`",
        f"- Benchmark SHA-256: `{results['benchmark_metadata']['sha256']}`",
        f"- Embedding model: `{results['configuration']['embedding_model']}`",
        f"- Top-k: {results['configuration']['top_k']}",
        f"- Agentic similarity threshold: {results['configuration']['similarity_threshold']}",
        f"- Agentic maximum retrieval rounds: {results['configuration']['maximum_agentic_rounds']}",
        "",
        "## Aggregate retrieval metrics",
        "",
        "| Metric (macro-average over all queries) | Conventional RAG | Agentic RAG | Agentic - conventional |",
        "|---|---:|---:|---:|",
    ]
    for metric in conventional:
        lines.append(
            f"| {metric} | {conventional[metric]:.4f} | "
            f"{agentic[metric]:.4f} | "
            f"{summary['agentic_minus_conventional'][metric]:+.4f} |"
        )
    agent_metrics = summary["agentic_specific"]
    lines.extend(
        [
            "",
            "## Agentic retrieval behavior",
            "",
            f"- Average rounds per successful Agentic RAG run: {agent_metrics['average_retrieval_rounds']['value']:.4f} "
            f"(denominator: {agent_metrics['average_retrieval_rounds']['denominator']} successful runs)",
            f"- First-round sufficiency rate: {agent_metrics['first_round_sufficiency_rate']['value']:.4f} "
            f"(denominator: {agent_metrics['first_round_sufficiency_rate']['denominator']} benchmark queries)",
            f"- Second-round trigger rate: {agent_metrics['second_round_trigger_rate']['value']:.4f} "
            f"(denominator: {agent_metrics['second_round_trigger_rate']['denominator']} benchmark queries)",
            f"- Second-round recovery rate: {agent_metrics['second_round_recovery_rate']['value']} "
            f"(denominator: {agent_metrics['second_round_recovery_rate']['denominator']} first-round-insufficient runs)",
            f"- Final evidence sufficiency rate: {agent_metrics['evidence_sufficiency_rate']['value']:.4f} "
            f"(denominator: {agent_metrics['evidence_sufficiency_rate']['denominator']} benchmark queries)",
            f"- Query refinement rate: {agent_metrics['query_refinement_rate']['value']:.4f} "
            f"(denominator: {agent_metrics['query_refinement_rate']['denominator']} benchmark queries)",
            f"- Average final selected documents: {agent_metrics['average_final_documents']['value']:.4f} "
            f"(denominator: {agent_metrics['average_final_documents']['denominator']} successful runs)",
            f"- Retrieval errors: {len(results['errors'])}",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python scripts/run_rag_benchmark.py",
            "python -m pytest -q",
            "```",
            "",
            "This is a small manually labeled retrieval benchmark over the current "
            "corpus, not a clinical or melanoma-classification accuracy evaluation. "
            "The Agentic RAG planner accepts structured ViT/ABCDE/evolution inputs "
            "rather than an arbitrary user query; each benchmark query includes a "
            "fixed structured case input, and the live agent generates its own "
            "targeted queries. Relevance labels are based only on the cited corpus "
            "chunks. Interpret aggregate results cautiously because the corpus is "
            "small and the queries are not independent.",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    corpus_documents: list[dict[str, Any]] | None = None,
    conventional_retriever: Callable[[str, int], list[dict[str, Any]]] | None = None,
    agent_factory: Callable[..., Any] | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run both systems against the same fixed query set and current corpus."""
    if corpus_documents is None:
        from app.database import rag_documents_collection

        corpus_documents = list(
            rag_documents_collection.find(
                {},
                {
                    "_id": 1,
                    "source": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "embedding": 1,
                },
            )
        )
    if conventional_retriever is None:
        conventional_retriever = retrieve_documents
    if agent_factory is None:
        agent_factory = ClinicalRAGAgent

    benchmark, benchmark_sha256, natural_lookup = load_benchmark(
        dataset_path,
        corpus_documents,
    )
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    agent_logger = _WarningCollector()
    agent_logger.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    AGENT_LOGGER.addHandler(agent_logger)
    try:
        agent = agent_factory(
            max_rounds=MAX_RETRIEVAL_ROUNDS,
            relevance_threshold=RELEVANCE_THRESHOLD,
            top_k=TOP_K,
        )
        for item in benchmark["queries"]:
            query_id = item["query_id"]
            relevant_ids = set(item["relevant_documents"])
            grades = item["relevance_grades"]
            record: dict[str, Any] = {
                "query_id": query_id,
                "category": item["category"],
                "query": item["query"],
                "relevant_documents": item["relevant_documents"],
                "relevance_grades": grades,
                "conventional_documents": [],
                "agentic_rounds": [],
                "agentic_final_documents": [],
                "agentic_provenance": [],
                "agentic_trace": None,
                "round1_sufficient": None,
                "final_sufficient": False,
                "rounds_used": 0,
                "second_round_triggered": False,
                "query_refined": False,
                "conventional_error": None,
                "agentic_error": None,
                "agentic_warnings": [],
            }

            try:
                conventional_docs = conventional_retriever(item["query"], TOP_K)
                record["conventional_documents"] = [
                    _result_identity(doc, natural_lookup)
                    for doc in conventional_docs
                ]
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record["conventional_error"] = message
                errors.append(
                    {"query_id": query_id, "system": "conventional", "error": message}
                )

            agent_logger.messages.clear()
            try:
                agent_result = agent.run(
                    classification=item["agentic_input"]["classification"],
                    abcde=item["agentic_input"]["abcde"],
                    evolution=item["agentic_input"]["evolution"],
                    top_k=TOP_K,
                )
                trace = agent_result.get("agent_trace", {})
                rounds = trace.get("rounds", [])
                record["agentic_trace"] = trace
                record["agentic_provenance"] = agent_result.get("provenance", [])
                record["rounds_used"] = int(agent_result.get("rounds_executed", len(rounds)))
                record["second_round_triggered"] = len(rounds) >= 2
                if rounds:
                    record["round1_sufficient"] = not rounds[0].get(
                        "missing_domains", []
                    )
                record["final_sufficient"] = (
                    agent_result.get("sufficiency") == "sufficient"
                )
                record["agentic_rounds"] = [
                    {
                        "iteration": round_data.get("iteration"),
                        "queries": round_data.get("queries", []),
                        "retrieved_documents": [
                            _result_identity(document, natural_lookup)
                            for document in round_data.get(
                                "retrieved_documents", []
                            )
                        ],
                        "similarity_scores": round_data.get(
                            "similarity_scores", []
                        ),
                        "missing_domains": round_data.get("missing_domains", []),
                        "evidence_sufficient": not round_data.get(
                            "missing_domains", []
                        ),
                        "decision": round_data.get("decision"),
                    }
                    for round_data in rounds
                ]
                record["agentic_final_documents"] = [
                    _result_identity(document, natural_lookup)
                    for document in agent_result.get("documents", [])
                ]
                if len(rounds) >= 2:
                    initial_queries = set(rounds[0].get("queries", []))
                    refined_queries = rounds[1].get("queries", [])
                    record["query_refined"] = any(
                        query not in initial_queries for query in refined_queries
                    )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record["agentic_error"] = message
                errors.append({"query_id": query_id, "system": "agentic", "error": message})
            finally:
                record["agentic_warnings"] = list(agent_logger.messages)
                for warning in record["agentic_warnings"]:
                    errors.append(
                        {
                            "query_id": query_id,
                            "system": "agentic_warning",
                            "error": warning,
                        }
                    )

            conventional_ids = [
                document["document_id"]
                for document in record["conventional_documents"]
            ]
            agentic_ids = [
                document["document_id"]
                for document in record["agentic_final_documents"]
            ]
            conventional_metrics = retrieval_metrics(
                conventional_ids, relevant_ids, grades
            )
            agentic_metrics = retrieval_metrics(agentic_ids, relevant_ids, grades)
            for name, value in conventional_metrics.items():
                record[f"conventional_{name}"] = value
            for name, value in agentic_metrics.items():
                record[f"agentic_{name}"] = value
            records.append(record)
    finally:
        AGENT_LOGGER.removeHandler(agent_logger)

    conventional_summary = _aggregate(records, "conventional")
    agentic_summary = _aggregate(records, "agentic")
    metric_deltas = {
        name: agentic_summary[name] - conventional_summary[name]
        for name in conventional_summary
    }
    successful_agentic = [row for row in records if row["agentic_error"] is None]
    recovery_metrics = second_round_recovery_rate(records)
    total_queries = len(records)
    n_triggered = sum(row["second_round_triggered"] for row in records)
    agentic_specific = {
        "average_retrieval_rounds": {
            "value": _mean([float(row["rounds_used"]) for row in successful_agentic]),
            "numerator": sum(row["rounds_used"] for row in successful_agentic),
            "denominator": len(successful_agentic),
        },
        "first_round_sufficiency_rate": {
            "value": sum(row["round1_sufficient"] is True for row in records)
            / total_queries,
            "numerator": sum(row["round1_sufficient"] is True for row in records),
            "denominator": total_queries,
        },
        "second_round_trigger_rate": {
            "value": n_triggered / total_queries,
            "numerator": n_triggered,
            "denominator": total_queries,
        },
        "second_round_recovery_rate": {
            **recovery_metrics,
        },
        "evidence_sufficiency_rate": {
            "value": sum(row["final_sufficient"] for row in records) / total_queries,
            "numerator": sum(row["final_sufficient"] for row in records),
            "denominator": total_queries,
        },
        "query_refinement_rate": {
            "value": sum(row["query_refined"] for row in records) / total_queries,
            "numerator": sum(row["query_refined"] for row in records),
            "denominator": total_queries,
        },
        "average_final_documents": {
            "value": _mean(
                [float(len(row["agentic_final_documents"])) for row in successful_agentic]
            ),
            "numerator": sum(
                len(row["agentic_final_documents"]) for row in successful_agentic
            ),
            "denominator": len(successful_agentic),
        },
        "average_unique_documents_across_rounds": {
            "value": _mean(
                [
                    float(
                        len(
                            {
                                document["document_id"]
                                for round_data in row["agentic_rounds"]
                                for document in round_data["retrieved_documents"]
                            }
                        )
                    )
                    for row in successful_agentic
                ]
            ),
            "denominator": len(successful_agentic),
        },
    }

    corpus_hash = corpus_fingerprint(corpus_documents)
    sources = sorted({str(document.get("source")) for document in corpus_documents})
    results = {
        "benchmark_metadata": {
            "benchmark_id": benchmark["benchmark_id"],
            "query_count": len(records),
            "sha256": benchmark_sha256,
            "id_scheme": benchmark["id_scheme"],
            "labeling_method": "Manual relevance labels from stored corpus text; grades: 3 direct, 2 substantial, 1 partial relevance.",
            "fixed_random_seed": RANDOM_SEED,
        },
        "corpus_metadata": {
            "database": "cefm_melanoma",
            "collection": "rag_documents",
            "document_count": len(corpus_documents),
            "sources": sources,
            "sha256": corpus_hash,
        },
        "configuration": {
            "embedding_model": MODEL_NAME,
            "similarity_metric": "cosine_similarity (existing app.rag.retrieval implementation)",
            "top_k": TOP_K,
            "similarity_threshold": RELEVANCE_THRESHOLD,
            "maximum_agentic_rounds": MAX_RETRIEVAL_ROUNDS,
            "agentic_input_contract": (
                "The actual ClinicalRAGAgent accepts structured classification, "
                "ABCDE, and evolution inputs and generates targeted queries; it "
                "does not accept the benchmark natural-language query directly."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sentence_transformers": _package_version("sentence-transformers"),
            "pymongo": _package_version("pymongo"),
        },
        "summary": {
            "conventional": conventional_summary,
            "agentic": agentic_summary,
            "agentic_minus_conventional": metric_deltas,
            "agentic_specific": agentic_specific,
            "denominator_policy": (
                "Retrieval metric macro-averages include every benchmark query; "
                "a retrieval error contributes an empty ranking. Agentic average "
                "rounds/document counts use successful Agentic RAG runs. Sufficiency "
                "and trigger/refinement rates use all benchmark queries. Recovery "
                "uses only observed first-round-insufficient runs; it is null if "
                "there are no such runs."
            ),
        },
        "per_query_results": records,
        "errors": errors,
        "limitations": [
            "The current live MongoDB corpus contains only 14 chunks from one source file.",
            "This fixed benchmark has fewer than 40 items because the current corpus is small.",
            "Relevance judgments are manual, corpus-grounded annotations rather than independent expert adjudication.",
            "The Agentic RAG interface is structured-case driven; the fixed case inputs encode each query's intent, and the agent produces its own targeted query strings.",
            "This measures retrieval relevance and evidence sufficiency only, not clinical or melanoma classification accuracy.",
        ],
    }
    if write_outputs:
        _write_outputs(output_dir, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark conventional and Agentic RAG against the fixed current corpus."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_benchmark(args.dataset, args.output_dir)
    print(f"Queries: {result['benchmark_metadata']['query_count']}")
    print(f"Corpus documents: {result['corpus_metadata']['document_count']}")
    for system in ("conventional", "agentic"):
        metrics = result["summary"][system]
        print(
            f"{system}: P@1={metrics['precision_at_1']:.4f}, "
            f"P@3={metrics['precision_at_3']:.4f}, "
            f"P@5={metrics['precision_at_5']:.4f}, "
            f"MRR={metrics['mrr']:.4f}, "
            f"NDCG@5={metrics['ndcg_at_5']:.4f}"
        )
    print(f"Errors: {len(result['errors'])}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
