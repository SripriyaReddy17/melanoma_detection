# Agentic RAG Implementation Report
**Project**: An Agentic Retrieval-Augmented Framework for Explainable Melanoma Diagnosis using Clinical Features  
**Root Path**: `D:\project 3-1\final_version`  
**Phase**: Implementation & Verification  
**Date**: October 2026  

---

## 1. Files Created

| File Path | Description |
|---|---|
| [`app/rag/agentic/__init__.py`](file:///d:/project%203-1/final_version/app/rag/agentic/__init__.py) | Package initialization exporting `ClinicalRAGAgent`, `EvidenceEvaluator`, and bounded configuration constants. |
| [`app/rag/agentic/clinical_agent.py`](file:///d:/project%203-1/final_version/app/rag/agentic/clinical_agent.py) | Core `ClinicalRAGAgent` controller implementing the bounded multi-round retrieval loop, targeted query planning, provenance tracking, and structured audit trace generation. |
| [`app/rag/agentic/evaluator.py`](file:///d:/project%203-1/final_version/app/rag/agentic/evaluator.py) | `EvidenceEvaluator` class evaluating document cosine similarity against the engineering threshold (`RELEVANCE_THRESHOLD = 0.45`) and matching domain-specific clinical terminology. |
| [`tests/test_agentic_rag.py`](file:///d:/project%203-1/final_version/tests/test_agentic_rag.py) | Comprehensive 15-test unit suite covering initialization, query targeting, feature preservation, safety bounds, and trace generation without requiring an active MongoDB connection. |
| [`scripts/evaluate_agentic_rag.py`](file:///d:/project%203-1/final_version/scripts/evaluate_agentic_rag.py) | Evaluation benchmark framework for Precision@K, Recall@K, MRR, NDCG@K, sufficiency rate, and round statistics, with reporting when labeled ground truth is unavailable. |

---

## 2. Files Modified

| File Path | Changes Made | Rationale |
|---|---|---|
| [`app/rag/__init__.py`](file:///d:/project%203-1/final_version/app/rag/__init__.py) | Exported `ClinicalRAGAgent` alongside legacy `retrieve_melanoma_evidence`. | Preserves backward compatibility while exposing the new agentic interface. |
| [`app/services/analysis.py`](file:///d:/project%203-1/final_version/app/services/analysis.py) | Integrated `ClinicalRAGAgent` into Stage 13 with automatic fallback to conventional `retrieve_melanoma_evidence`. Added structured `agentic_rag` metadata to pipeline output. | Replaces static monolithic RAG with dynamic bounded Agentic RAG while maintaining strict contract compatibility for reports and callers. |

---

## 3. Current Conventional RAG Architecture

Before this implementation, the RAG architecture operated as a single-turn static retrieval process:
```
Dermoscopic Image
       ↓
ViT + ABCDE Pipeline
       ↓
Monolithic Query Generation (build_melanoma_query)
  - Concatenates predicted class, prob, A, B, C, D, E into one text block
       ↓
Dense Embedding (create_embedding: all-MiniLM-L6-v2, 384-D)
       ↓
Linear Scan & Cosine Similarity in MongoDB (retrieve_documents)
       ↓
Top-3 Documents Dumped Directly into Report (no sufficiency check)
```

### Limitations of Static RAG:
1. **Semantic Dilution**: A 12-line monolithic prompt mixes multiple clinical dimensions, diluting specific vector matching for high-risk anomalies.
2. **Zero Verification**: Top-3 documents were attached regardless of whether similarity scores were clinically meaningful or whether the retrieved text actually discussed the patient's dominant risk feature.
3. **No Dynamic Follow-up**: If initial retrieval missed guidelines for a dominant anomaly, the system had no mechanism to formulate gap queries.

---

## 4. New Agentic RAG Architecture

The newly implemented Agentic RAG architecture introduces an autonomous, self-evaluating clinical evidence retrieval and synthesis loop:

```
                  +-----------------------------------------+
                  |            Observed Findings            |
                  | (Classification, ABCDE, ViT Prob, Risk) |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |            ClinicalRAGAgent             |
                  |          (app.rag.agentic)              |
                  +-----------------------------------------+
                         |                     ^
            1. Formulate |                     | 4. Return Evaluation
            Targeted     |                     |    & Sufficiency Check
            Queries      v                     |
                  +-----------------+   +--------------------+
                  |  Query Planner  |   | Evidence Evaluator |
                  |  (Domain-based  |   | (Sufficiency & Gap |
                  |   Rules)        |   |  Analysis)         |
                  +-----------------+   +--------------------+
                         |                     ^
            2. Dense     |                     | 3. Ranked Chunks &
               Embedding v                     |    Similarity Scores
                  +-----------------------------------------+
                  |         Existing RAG Retriever          |
                  | (app.rag.retrieval: retrieve_documents) |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |         MongoDB Knowledge Base          |
                  |        (cefm_melanoma database)         |
                  +-----------------------------------------+
                                       |
               [Loop terminates if Sufficient OR Iteration >= 2]
                                       |
                                       v
                  +-----------------------------------------+
                  |       Structured Evidence Summary       |
                  | (4-Tier Provenance: OBSERVED, ML MODEL, |
                  |  RETRIEVED, UNAVAILABLE + Agent Trace)  |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |        Existing Report Generator        |
                  |   (app/services/report_generation.py)   |
                  +-----------------------------------------+
```

---

## 5. Agent Decision Process

The agent operates strictly deterministically without non-deterministic LLM hallucination:
1. **Target Identification**:
   - Classification domain: Formulated based on ViT predicted class and melanoma probability ($\ge 0.50$ targets malignant melanoma excision and radial/vertical growth; $< 0.50$ targets benign melanocytic stability).
   - Asymmetry domain: Formulated only when asymmetry score $\ge 0.35$.
   - Border domain: Formulated only when border irregularity score $\ge 0.35$.
   - Color domain: Formulated only when color variation index $\ge 0.30$.
   - Diameter domain: Formulated only when calibrated physical measurement is present.
   - Evolution domain: Formulated only when prior comparison images exist.
2. **Evidence Evaluation**:
   - Retrieved chunks are evaluated by `EvidenceEvaluator`.
   - Each target domain must be covered by at least one chunk with similarity score $\ge$ `RELEVANCE_THRESHOLD` (0.45) containing domain clinical keywords.
3. **Termination vs. Retry Decision**:
   - If all target domains are covered: Decision = `"stop"`, sufficiency = `"sufficient"`.
   - If any domain lacks sufficient evidence and `iteration < 2`: Decision = `"retrieve_again"`.
   - If `iteration == 2`: Decision = `"stop"` (hard bounded limit).

---

## 6. Retrieval Loop

1. **Round 1**:
   - Plans initial targeted queries based on observed findings.
   - Calls existing `retrieve_documents()` for each query.
   - Deduplicates retrieved chunks by source, chunk index, and content while retaining maximal similarity score.
   - Evaluates coverage against target domains.
2. **Round 2 (Gap Resolution)**:
   - If evidence is insufficient, identifies `missing_domains`.
   - Generates focused gap queries specifically targeting the missing domains using refined clinical search terms.
   - Retrieves and re-evaluates combined evidence pool.
   - Strictly terminates after Round 2 (`MAX_RETRIEVAL_ROUNDS = 2`).

---

## 7. Evidence Provenance

To prevent diagnostic over-reach and clinical hallucination, every piece of evidence is partitioned into four distinct tiers:

1. `[OBSERVED]`: Directly measured physical/morphological values produced by the deterministic image pipeline:
   - Lesion asymmetry score, border irregularity score, color index, pixel extents, or calibrated mm sizes.
2. `[ML MODEL]`: Inferences output by neural network architectures:
   - Vision Transformer (ViT) predicted class, melanoma probability percentage, risk category.
3. `[RETRIEVED]`: Verified medical guidelines and clinical literature obtained verbatim from the MongoDB knowledge base.
4. `[UNAVAILABLE]`: Explicit declarations of missing clinical variables:
   - Uncalibrated diameter (when no ruler/marker is present).
   - Evolution history (when no prior visit baseline image was provided). The system strictly refuses to extrapolate evolution from a single static image.

---

## 8. Safety Constraints

- **Prediction Invariance**: The agent is strictly forbidden from altering ViT predictions, probabilities, or ABCDE numerical values.
- **Strict Bounded Execution**: The agent cannot enter an infinite or open-ended retrieval loop (`MAX_RETRIEVAL_ROUNDS = 2`).
- **Engineering Cutoff Discipline**: `RELEVANCE_THRESHOLD = 0.45` is explicitly defined and documented as an engineering configuration, not an asserted clinical standard.
- **Fail-Safe Fallback**: If the agent encounters an unexpected exception, the system automatically falls back to conventional RAG and image analysis to prevent pipeline disruption.

---

## 9. Test Results

### Unit Test Suite (`tests/test_agentic_rag.py`)
All 15 targeted tests passed:
- `test_1_agent_initializes`: PASSED
- `test_2_targeted_queries_are_generated`: PASSED
- `test_3_asymmetry_query_generation`: PASSED
- `test_4_border_query_generation`: PASSED
- `test_5_color_query_generation`: PASSED
- `test_6_diameter_query_generation`: PASSED
- `test_7_evolution_is_not_fabricated`: PASSED
- `test_8_existing_retriever_is_called`: PASSED
- `test_9_sufficient_evidence_stops_retrieval`: PASSED
- `test_10_insufficient_evidence_triggers_second_round`: PASSED
- `test_11_maximum_two_rounds_enforced`: PASSED
- `test_12_agent_trace_is_generated`: PASSED
- `test_13_model_prediction_remains_unchanged`: PASSED
- `test_14_unavailable_information_remains_unavailable`: PASSED
- `test_15_empty_retrieval_handled_safely`: PASSED

### Full Regression Suite (`python -m pytest -q`)
- **Total Tests**: 41 passed (26 existing + 15 new)
- **Failures**: 0
- **Execution Time**: ~19 seconds

### Conventional RAG Compatibility (`test_rag.py`)
- Executed successfully against active MongoDB knowledge base.
- Retrieved top-3 chunks from `melanoma_abcde.pdf` with identical behavior.

---

## 10. Evaluation Limitations

The evaluation script [`scripts/evaluate_agentic_rag.py`](file:///d:/project%203-1/final_version/scripts/evaluate_agentic_rag.py) implements the mathematical formulations for:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Evidence Sufficiency Rate
- Average Retrieval Rounds

When executed in the absence of a labeled ground-truth dataset, the script accurately reports:
```
Evaluation dataset not available.
File not found: data\evaluation\rag_benchmark.json
Note: Quantitative retrieval benchmarking requires an expert-annotated clinical ground-truth dataset. Quantitative Agentic RAG evaluation remains future work.
```
No synthetic or fabricated benchmark scores were generated.

---

## 11. Example Agent Trace

From a live execution with real pipeline inputs (`melanoma_probability = 0.82`, `asymmetry = 0.68`, `border = 0.55`, `color = 0.42`, `evolution unavailable`):

```json
{
  "agent_type": "clinical_rag_agent",
  "max_rounds": 2,
  "rounds": [
    {
      "iteration": 1,
      "queries": [
        "Cutaneous malignant melanoma clinical presentation, radial vs vertical growth phase, and surgical excision guidelines.",
        "Dermoscopic asymmetry along orthogonal axes and differential melanocytic pigment distribution.",
        "Irregular notched borders, margin abrupt termination, and peripheral contour roughness in dermoscopy.",
        "Color variegation, multi-chromatic pigment distribution, and blue-gray or dark brown melanoma areas."
      ],
      "retrieved_documents": [
        {
          "source": "melanoma_abcde.pdf",
          "chunk_index": 4,
          "score": 0.5889
        },
        {
          "source": "melanoma_abcde.pdf",
          "chunk_index": 6,
          "score": 0.5805
        },
        {
          "source": "melanoma_abcde.pdf",
          "chunk_index": 5,
          "score": 0.5895
        },
        {
          "source": "melanoma_abcde.pdf",
          "chunk_index": 2,
          "score": 0.3476
        }
      ],
      "similarity_scores": [
        0.5889,
        0.5805,
        0.5895,
        0.3476
      ],
      "missing_domains": [],
      "decision": "stop"
    }
  ]
}
```

---

## 12. Remaining Work

1. **Expert Labeled Benchmark Dataset**: Annotate a standardized set of 50 dermoscopic cases with ground-truth literature chunk mappings (`data/evaluation/rag_benchmark.json`) to compute quantitative IR metrics (Precision@K, Recall@K, NDCG@K).
2. **Clinical Expert Feedback**: Present the generated 4-tier provenance sections to practicing dermatologists for qualitative validation of explainability impact.
