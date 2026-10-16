# Agentic RAG Architecture & Design Specification
**Project**: An Agentic Retrieval-Augmented Framework for Explainable Melanoma Diagnosis using Clinical Features  
**Root Path**: `D:\project 3-1\final_version`  
**Phase**: Historical analysis/design specification (implementation is now present)  
**Date**: October 2026  

> **Current runtime note:** This document records the earlier design baseline
> and proposal; it is not the authoritative description of the active
> application pipeline. Active flow is Input Image → Image Validation →
> Calibration → UltraLight VM-UNet → ABCDE Feature Extraction → ViT
> Classification → Agentic RAG → Report Generation. See `PROJECT_STATUS.md`.
> The implementation exists in `app/rag/agentic/`; quantitative Agentic RAG
> benchmarking remains pending.

---

## 1. Historical RAG Baseline (At Design Time)

At the time of this design inspection, the RAG implementation in `app/rag/`
was a **single-turn, static retrieval system**. The following flow and
limitations describe that historical baseline, not the active implementation.

### Historical RAG Data Flow
```
Dermoscopic Image Input
         ↓
Image Processing & Feature Extraction
  - Lesion Segmentation (app/services/segmentation.py)
  - ABCDE Clinical Extraction (app/services/analysis.py)
  - ViT Classification (app/services/classification.py)
         ↓
Monolithic Query Construction (app/rag/rag_service.py: build_melanoma_query)
         ↓
Vector Embedding Generation (app/rag/embeddings.py: create_embedding)
  - sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vector)
         ↓
Exhaustive MongoDB Search & Python Cosine Scoring (app/rag/retrieval.py: retrieve_documents)
  - Queries collection 'rag_documents' in database 'cefm_melanoma'
  - Computes dot product / vector norms across all stored chunks
         ↓
Top-K Ranking & Extraction (top_k = 3)
         ↓
Direct Ingestion into Structured Report (app/services/report_generation.py: generate_structured_report)
         ↓
Fact-Checking Validation (app/services/report_fact_checker.py: validate_report)
```

### Limitations of the Historical Static RAG
1. **Monolithic Query Distortion**: It combines predicted class, probability, asymmetry, border irregularity, color index, diameter, and evolution into one 12-line query string. Semantic embeddings of long multi-topic strings suffer from topic dilution, often failing to retrieve specific clinical guidelines for a patient's dominant risk feature (e.g., severe border notched margins vs. uncalibrated diameter).
2. **Single-Shot Failure Mode**: If top-3 retrieved documents do not address the lesion's specific clinical anomalies (e.g., a high asymmetry score with zero matched documents on geometric reflection), the system has no mechanism to critique its own retrieval, identify evidence gaps, or refine its search query.
3. **No Evidence Sufficiency Evaluation**: Top-k documents are included regardless of whether their similarity scores meet clinical relevance standards or whether they address the actual diagnostic questions posed by the classifier.
4. **Lack of Information Provenance**: Retrieved text is appended as raw paragraphs without delineating what was observed in the image versus what clinical literature recommends.

---

## 2. Current RAG API Inventory

The exact functions and classes responsible for the RAG pipeline are:

### A. Creating Embeddings
- **File**: `app/rag/embeddings.py`
- **Function**: `create_embedding(text: str) -> list[float]`
- **Input**: `text: str` (query string or document chunk)
- **Output**: `list[float]` (384-dimensional normalized vector)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Called by**:
  - `app/rag/retrieval.py: retrieve_documents()`
  - `app/rag/ingest.py: ingest_pdf()`

### B. Query Construction
- **File**: `app/rag/rag_service.py`
- **Function**: `build_melanoma_query(classification: dict, abcde: dict, evolution: dict) -> str`
- **Input**:
  - `classification: dict` (`predicted_class`, `melanoma_probability`)
  - `abcde: dict` (`asymmetry`, `border_irregularity`, `color_index`, `diameter_mm`)
  - `evolution: dict` (`change_detected`)
- **Output**: Formatted multi-line clinical prompt string
- **Called by**: `app/rag/rag_service.py: retrieve_melanoma_evidence()`

### C. Retrieving & Ranking Documents
- **File**: `app/rag/retrieval.py`
- **Function**: `retrieve_documents(query: str, top_k: int = 3) -> list[dict]`
- **Input**:
  - `query: str`
  - `top_k: int` (default: 3)
- **Output**: Sorted list of dictionaries:
  ```python
  [
      {
          "source": str,        # e.g., "melanoma_abcde.pdf"
          "chunk_index": int,   # e.g., 6
          "text": str,          # Chunk text content
          "score": float        # Cosine similarity score rounded to 4 decimals
      },
      ...
  ]
  ```
- **Helper Function**: `cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float`
- **Called by**: `app/rag/rag_service.py: retrieve_melanoma_evidence()`

### D. Returning Retrieved Context
- **File**: `app/rag/rag_service.py`
- **Function**: `retrieve_melanoma_evidence(classification: dict, abcde: dict, evolution: dict, top_k: int = 3) -> dict`
- **Input**:
  - `classification: dict`
  - `abcde: dict`
  - `evolution: dict`
  - `top_k: int`
- **Output**:
  ```python
  {
      "query": str,
      "documents": list[dict]
  }
  ```
- **Called by**:
  - `app/services/analysis.py` (line 479)
  - `test_rag.py` (line 25)

### E. Generating the Final Report
- **File**: `app/services/report_generation.py`
- **Function**: `generate_structured_report(validation, classification, abcde, evolution, explanation, rag_evidence) -> dict`
- **Input**: Pipeline state dictionaries including `rag_evidence`
- **Output**: Comprehensive assessment dictionary containing `full_report_text`, structured clinical sections, formatted RAG references, and clinical disclaimers
- **Called by**: `app/services/analysis.py` (line 506)

---

## 3. Inspection of Current Knowledge Base & Database

### Clinical Documents on Disk
Located in `data/medical_docs/`:
1. `AAD_ABCDE_Melanoma.pdf` (1,700,668 bytes) - American Academy of Dermatology guidelines on ABCD criteria.
2. `Melanoma Treatment - NCI.pdf` (1,189,298 bytes) - National Cancer Institute clinical treatment overview.
3. `NCI.pdf` (148,899 bytes) - National Cancer Institute melanoma detection manual.
4. `melanoma_abcde.pdf` (1,867,112 bytes) - Patient and clinical guide on melanoma warning signs and self-examination.

### Curated Knowledge Base (Pre-structured)
Located in `app/rag/agentic/knowledge_base.py`:
- 8 peer-reviewed knowledge units (`KB-MEL-001` through `KB-LIMIT-008`).
- Metadata fields: `id`, `topic`, `feature`, `source_type`, `year`, `reliability`, `title`, `text`.
- Covers melanoma overview, benign nevi, asymmetry criteria, border irregularity, color variegation, diameter calibration standards, longitudinal evolution, and AI clinical limitations.

### Active Database & Collection State
- **Database Engine**: MongoDB (local instance `mongodb://localhost:27017`)
- **Database Name**: `cefm_melanoma`
- **Collection**: `rag_documents`
- **Current Ingested Chunks**: 14 chunks currently present (source: `melanoma_abcde.pdf`).
- **Chunk Parameters**: `CHUNK_SIZE = 1000` characters, `CHUNK_OVERLAP = 150` characters (`app/rag/ingest.py`).
- **Storage Schema**:
  ```json
  {
    "_id": ObjectId("..."),
    "source": "melanoma_abcde.pdf",
    "chunk_index": 6,
    "text": "...",
    "embedding": [0.0412, -0.0128, ...] // 384 dimensions
  }
  ```

---

## 4. Current Melanoma Pipeline & Inter-Component Contracts

In `app/services/analysis.py: analyze_image()`, the data structures passed between stages are strictly defined:

| Pipeline Stage | Function Called | Input Object | Output Object Produced |
|---|---|---|---|
| Image Validation | `validate_image_quality()` | Raw file path, bytes | `validation: dict` (`is_valid`, `reasons`, `dimensions`, `dpi`, `metrics`) |
| Marker Calibration | `calibrate_image()` | BGR image `np.ndarray` | `calibration: dict` (`success`, `pixels_per_mm`, `confidence`, `detected_features`) |
| Lesion Segmentation | `segment_lesion()` | Image path | `mask: np.ndarray` (uint8 binary mask, 0 or 1) |
| Asymmetry (A) | `calculate_asymmetry()` | Binary `mask` | `asymmetry: dict` (`score`: 0.0–1.0, `horizontal_asymmetry`, `vertical_asymmetry`, `severity`) |
| Border (B) | `calculate_border_irregularity()` | Binary `mask` | `border: dict` (`border_irregularity_score`: 0.0–1.0, `circularity`, `compactness`, `curvature_variation`) |
| Color (C) | `calculate_color_variation()` | RGB image + `mask` | `color: dict` (`index`: 0.0–1.0, `h`, `s`, `v`, `hue_variation`, `saturation_variation`, `brightness_variation`, `dominant_colors`) |
| Diameter (D) | `calculate_diameter()` | Binary `mask`, `pixels_per_mm` | `diameter: dict` (`max_diameter_pixels`, `max_diameter_mm`, `calibration_used`: bool, `available`: bool) |
| Evolution (E) | `analyze_evolution()` | Current + prior features | `evolution: dict` (`available`: bool, `change_detected`: bool, `overall_change`: float, `changes`: dict) |
| ViT Classification | `classify()` | Image path, `features` | `classification: dict` (`predicted_class`: "mel"/"non_mel", `melanoma_probability`: float, `probabilities`: dict, `risk_level`: "low"/"medium"/"high", `backbone`, `model_version`) |
| Contrastive Alignment | `contrastive_aligner.align()` | Image path, ABCDE vector | `contrastive: dict` (`score`: float, `status`, `latent_dim`: 128) |
| CLIP Concepts | `clip_extractor.extract_concepts()` | Image path, `mask`, ABCDE | `clip: dict` (`concepts`: list of ranked visual concept dicts, `scores`) |
| Flattened Clinical Features | Internal flattening in `analysis.py` | Dicts A, B, C, D, E | `flat: dict` (`asymmetry`, `border_irregularity`, `color_index`, `diameter_mm`, `diameter_pixels`, `evolution_change_detected`) |

---

## 5. Agentic RAG Architecture Design

The proposed Agentic RAG system transforms static keyword/monolithic vector matching into an **autonomous clinical evidence retrieval and synthesis loop**.

### Conceptual Architecture
```
              +-----------------------------------------+
              |           Clinical Findings             |
              | (Classification, ABCDE, ViT Prob, Risk) |
              +-----------------------------------------+
                                   |
                                   v
              +-----------------------------------------+
              |           ClinicalRAGAgent              |
              |          (Agent State & Loop)           |
              +-----------------------------------------+
                     |                     ^
        1. Formulate |                     | 4. Return Evaluation
        Targeted     |                     |    & Sufficiency Check
        Query        v                     |
              +-----------------+   +--------------------+
              |  Query Planner  |   | Evidence Evaluator |
              |  (Deterministic |   | (Sufficiency & Gap |
              |   Clinical Rules)   |  Analysis)         |
              +-----------------+   +--------------------+
                     |                     ^
        2. Vectorized|                     | 3. Raw Ranked
           Query     v                     |    Documents
              +-----------------------------------------+
              |       Existing RAG Retriever            |
              | (app.rag.retrieval: retrieve_documents) |
              +-----------------------------------------+
                                   |
                                   v
              +-----------------------------------------+
              |        MongoDB Knowledge Base           |
              |  ('rag_documents' collection: 384-D)    |
              +-----------------------------------------+
                                   |
              [Loop terminates if Sufficient or Iteration >= 2]
                                   |
                                   v
              +-----------------------------------------+
              |      Structured Evidence Summary        |
              | (Provenanced Findings: Observed vs Lit) |
              +-----------------------------------------+
                                   |
                                   v
              +-----------------------------------------+
              |        Existing Report Generator        |
              |   (app/services/report_generation.py)   |
              +-----------------------------------------+
```

### Core Operating Principles
1. **Delegation to Existing Core**: The Agent does NOT reimplement vector search, embedding models, or MongoDB connectors. It directly invokes `app.rag.retrieval.retrieve_documents`.
2. **Deterministic Bounded Execution**: The loop is strictly bounded by `MAX_RETRIEVAL_ROUNDS = 2` (or 3 maximum). No infinite or open-ended execution.
3. **Targeted Clinical Queries**: Instead of 1 monolithic prompt, the agent constructs targeted clinical queries focused on anomalous features (e.g., if Asymmetry is 0.72, query specifically for asymmetrical clonal growth in dermatoscopy).
4. **Sufficiency & Gap Analysis**: After round 1, the agent inspects retrieved evidence:
   - Does retrieved text address the primary risk factors (melanoma probability / dominant ABCDE anomalies)?
   - Are similarity scores above the clinical threshold ($\ge 0.40$)?
   - If a dominant anomaly lacks grounded literature, round 2 generates a targeted fallback query specifically addressing the missing clinical domain.
5. **No Medical Fabrication**: The agent strictly grounds its reasoning in the retrieved documents. If no document is found for a feature (e.g., evolution unrecorded), it marks the feature as `UNAVAILABLE` rather than extrapolating.

---

## 6. Proposed Agent State Schema

The Agent State tracks every step of the retrieval trajectory for clinical explainability and scientific auditing:

```python
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class EvidenceItem:
    """Represents a single verified piece of retrieved medical evidence."""
    source: str
    chunk_index: int
    text: str
    similarity_score: float
    target_feature: str         # "classification", "A", "B", "C", "D", "E", "safety"
    retrieval_round: int

@dataclass
class RetrievalIterationTrace:
    """Internal reasoning and retrieval audit trace for a single round."""
    iteration: int
    queries_executed: List[str]
    retrieved_count: int
    highest_score: float
    evidence_sufficiency: str   # "sufficient", "insufficient", "partially_sufficient"
    gaps_identified: List[str]
    decision: str               # "retrieve_again", "terminate_success", "terminate_max_iterations"

@dataclass
class ClinicalAgentState:
    """Full execution state of the Agentic RAG pipeline."""
    # Inputs (Observed Findings)
    classification: Dict[str, Any] = field(default_factory=dict)
    abcde: Dict[str, Any] = field(default_factory=dict)
    evolution: Dict[str, Any] = field(default_factory=dict)
    
    # Internal Working Memory
    target_priorities: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    retrieved_evidence: List[EvidenceItem] = field(default_factory=list)
    traces: List[RetrievalIterationTrace] = field(default_factory=list)
    
    # Execution Bounds
    current_iteration: int = 1
    max_iterations: int = 2
    is_complete: bool = False
    
    # Final Structured Output
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

---

## 7. Agent Decision Logic & Query Planning Rules

The agent applies transparent, deterministic clinical rules to decide **what to query** and **when to stop**:

### Round 1: Targeted Query Generation
1. **Rule 1 (Classifier Alignment)**:
   - If `melanoma_probability >= 0.50` or `predicted_class == "mel"`:
     Query: `"Cutaneous malignant melanoma clinical presentation, radial vs vertical growth phase, and surgical excision guidelines."`
   - Else:
     Query: `"Benign melanocytic nevi dermoscopic stability, structural symmetry, and diagnostic monitoring criteria."`
2. **Rule 2 (Dominant ABCDE Feature Targeting)**:
   - Identify the most anomalous ABCDE features:
     - Asymmetry: `score >= 0.35` $\rightarrow$ Target `"A"`: `"Dermoscopic asymmetry along orthogonal axes and differential melanocytic pigment distribution."`
     - Border: `border_irregularity_score >= 0.35` $\rightarrow$ Target `"B"`: `"Irregular notched borders, margin abrupt termination, and peripheral contour roughness in dermoscopy."`
     - Color: `color_index >= 0.30` $\rightarrow$ Target `"C"`: `"Color variegation, multi-chromatic pigment distribution, and blue-gray or dark brown melanoma areas."`
     - Diameter: `diameter_mm >= 6.0` (with valid calibration) $\rightarrow$ Target `"D"`: `"Lesion diameter exceeding 6mm ABCDE guideline and physical calibration requirements."`
     - Evolution: `change_detected == True` $\rightarrow$ Target `"E"`: `"Longitudinal lesion evolution and temporal change detection across sequential dermoscopic visits."`
3. **Rule 3 (Negative Evidence / Missing Data Discipline)**:
   - If `evolution.available == False`: Do NOT search for lesion changes; record `evolution_status = "UNAVAILABLE"`.
   - If `calibration_used == False`: Do NOT search for millimeter size thresholds; record `diameter_physical_status = "UNAVAILABLE (pixel extent only)"`.

### Evidence Evaluation & Sufficiency Check
After receiving documents from `retrieve_documents()`, the agent checks:
- **Sufficiency Condition**:
  1. At least one document with `score >= 0.45` addresses the predicted class (melanoma vs. benign).
  2. For any feature flagged with High Risk (`asymmetry >= 0.50` or `border >= 0.50`), at least one document specifically mentions that clinical criterion.
- **Decision Logic**:
  - `IF (all clinical targets covered AND max_score >= 0.45) OR (iteration >= MAX_ROUNDS)`:
    - Set `decision = "terminate_success"`
    - Transition to Evidence Synthesis
  - `ELSE IF iteration < MAX_ROUNDS`:
    - Identify `gaps_identified` (e.g., `["asymmetry_criteria_missing"]`)
    - Set `decision = "retrieve_again"`
    - Proceed to Round 2 with focused gap queries

### Round 2: Refined Gap Querying (Fallback Round)
- In Round 2, the query is tightly focused only on the missing clinical domain with alternative dermatological keywords:
  - If Asymmetry missing: `"ABCD rule asymmetry criteria melanocytic lesion contour"`
  - If Border missing: `"Dermoscopy border notch structural irregularity margin"`
  - If Melanoma guideline missing: `"Dermatology guidelines cutaneous melanoma suspicious lesion"`

---

## 8. Medical Safety & Diagnostic Guardrails

The Agentic RAG system acts strictly as an **explainer and literature evidence retriever**, not an autonomous medical practitioner.

### Mandatory 4-Tier Information Partitioning
Every piece of generated output is tagged with its provenance to prevent hallucination:

1. `[OBSERVED]`: Directly measured from the patient's image:
   - Measured pixel/mm dimensions, asymmetry ratio, border contour curvature, HSV color standard deviations.
2. `[ML MODEL]`: Predictions produced by neural networks:
   - ViT predicted class (`mel` / `non_mel`), probability, risk band.
3. `[RETRIEVED]`: Medical facts fetched verbatim or cited directly from vetted literature:
   - Clinical guidelines, peer-reviewed study conclusions, AAD/NCI screening standards.
4. `[UNAVAILABLE]`: Explicit declaration of uncollected or unmeasured variables:
   - Uncalibrated physical size, unrecorded patient history, absent prior visit image.

### Non-Diagnostic Clinical Phrasing
- The agent **NEVER** outputs:
  - `"The patient has melanoma."`
  - `"This lesion is definitely benign."`
  - `"Prescription recommended."`
- The agent **ALWAYS** outputs calibrated decision-support language:
  - `"The vision transformer model identified visual patterns consistent with melanoma (probability: X%), supported by marked border irregularity."`
  - `"Clinical literature (AAD Guidelines) emphasizes that lesions exhibiting these morphological signs warrant formal dermoscopic evaluation by a dermatologist."`
  - `"Evolutionary history is unavailable because no prior examination image was submitted."`

---

## 9. Research Traceability & Audit Logs

To validate in a research publication that the framework is genuinely agentic (rather than a static script), the system produces a structured execution trace stored alongside the analysis:

### Example Research Audit Trace
```json
{
  "agent_name": "ClinicalRAGAgent",
  "version": "1.0.0",
  "total_rounds": 2,
  "execution_time_seconds": 1.42,
  "trace": [
    {
      "round": 1,
      "generated_queries": [
        "Cutaneous malignant melanoma clinical presentation and surgical excision guidelines.",
        "Dermoscopic significance of lesion asymmetry score 0.68 and mirror geometric imbalance."
      ],
      "retrieved_documents_count": 3,
      "top_score": 0.5824,
      "sufficiency_evaluation": {
        "status": "partially_sufficient",
        "covered_targets": ["classification_melanoma"],
        "uncovered_targets": ["asymmetry_clinical_rationale"]
      },
      "action": "TRIGGER_ROUND_2"
    },
    {
      "round": 2,
      "generated_queries": [
        "ABCD rule asymmetry criteria melanocytic lesion contour reflection mismatch"
      ],
      "retrieved_documents_count": 3,
      "top_score": 0.6102,
      "sufficiency_evaluation": {
        "status": "sufficient",
        "covered_targets": ["classification_melanoma", "asymmetry_clinical_rationale"],
        "uncovered_targets": []
      },
      "action": "TERMINATE_AND_SYNTHESIZE"
    }
  ],
  "final_retrieval_status": "verified_sufficient",
  "total_sources_cited": 2
}
```

---

## 10. Integration Architecture & Module Placement

### Recommended Component Location
The agent will be implemented as a clean, modular service in:
```
app/
  rag/
    agentic/
      clinical_agent.py      # Main ClinicalRAGAgent class & bounded loop
      evaluator.py           # Evidence sufficiency & gap detection logic
      state.py               # AnalysisState & ClinicalAgentState schemas
      knowledge_base.py      # Curated peer-reviewed knowledge units (already exists)
      planner.py             # Rule-based query planner (already exists)
      feature_agents.py      # ABCDE specialized evaluators (already exists)
```

### Clean Layering (Calling Existing Core)
```
app.services.analysis: analyze_image()
         ↓
app.rag.agentic.clinical_agent: run_clinical_rag_agent()
         ↓
Calls app.rag.retrieval: retrieve_documents()
         ↓
Interrogates MongoDB: cefm_melanoma.rag_documents
```

By placing the agent at `app/rag/agentic/clinical_agent.py`, the system:
1. Reuses `app/rag/retrieval.py` directly without duplicating vector similarity logic.
2. Leaves `app/rag/rag_service.py` intact for any legacy callers.
3. Allows `app/services/analysis.py` to simply swap `retrieve_melanoma_evidence()` with `run_clinical_rag_agent()` while receiving a backwards-compatible evidence payload.

---

## 11. Report Generation Integration

The existing report generation system (`app/services/report_generation.py: generate_structured_report`) expects:
```python
rag_evidence = {
    "query": str,
    "documents": list[dict]
}
```

To ensure zero breakage while introducing Agentic RAG:
1. The new `ClinicalRAGAgent` will produce a comprehensive dictionary that **strictly preserves the expected keys**:
   ```python
   agent_output = {
       # Backwards-compatible fields expected by report_generation.py
       "query": primary_query_text,
       "documents": best_deduplicated_documents,
       
       # Enhanced Agentic fields
       "agentic_metadata": {
           "rounds_executed": agent_state.current_iteration,
           "sufficiency": agent_state.sufficiency_status,
           "trace": agent_state.traces,
           "observed_vs_literature": agent_state.structured_provenance,
       }
   }
   ```
2. In `app/services/report_generation.py`, `_format_rag_evidence(rag_documents)` will continue to render the retrieved documents cleanly under `MEDICAL REFERENCE EVIDENCE (RAG)`.
3. An optional extension in `report_generation.py` can render the provenance badges (`[OBSERVED]`, `[RETRIEVED]`, `[ML MODEL]`, `[UNAVAILABLE]`) for richer explainability without disrupting existing tests.

---

## 12. Unit Test Plan

A dedicated test suite `tests/test_agentic_rag.py` will be created with 12 comprehensive unit tests:

1. `test_agent_initialization`: Verifies agent instantiates with default limits (`max_rounds=2`).
2. `test_agent_receives_valid_findings`: Checks that state correctly parses ViT classification and ABCDE inputs.
3. `test_agent_constructs_targeted_melanoma_query`: Confirms high-probability melanoma produces oncology-specific queries.
4. `test_agent_constructs_targeted_benign_query`: Confirms low-risk benign findings produce nevus-stability queries.
5. `test_agent_calls_existing_retriever`: Verifies `app.rag.retrieval.retrieve_documents` is invoked properly.
6. `test_agent_recognizes_sufficient_evidence`: Confirms agent terminates after Round 1 when top score exceeds threshold.
7. `test_agent_performs_second_round_on_gap`: Confirms agent executes Round 2 when initial evidence lacks target coverage.
8. `test_agent_strictly_respects_max_iterations`: Ensures agent never executes more than `max_iterations` rounds.
9. `test_agent_handles_zero_documents_safely`: Verifies pipeline gracefully handles empty MongoDB retrieval without crashing.
10. `test_agent_marks_evolution_unavailable`: Confirms missing prior image results in explicit `UNAVAILABLE` flag and no hallucinated temporal query.
11. `test_agent_never_overrides_vit_prediction`: Asserts that model classification and probability remain bitwise identical before and after RAG.
12. `test_agent_produces_structured_research_trace`: Confirms execution audit log contains iteration numbers, query texts, and sufficiency decisions.

---

## 13. Scientific Evaluation Plan for Research Publication

To substantiate the contribution in the research paper *"An Agentic Retrieval-Augmented Framework for Explainable Melanoma Diagnosis using Clinical Features"*, the following evaluation methodology will be deployed:

### Quantitative Retrieval Benchmark Metrics
- **Retrieval Precision@K ($K=3$)**: Fraction of retrieved chunks containing clinically accurate dermatology guidance for the query's target ABCDE feature.
- **Retrieval Recall@K**: Fraction of relevant clinical guidelines identified out of the total curated set.
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant clinical chunk.
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Ranking quality considering graduated relevance of clinical guidelines.

### Agentic Behavior Metrics
- **Evidence Sufficiency Rate**: Percentage of cases achieving `sufficient` status within 2 retrieval rounds.
- **Average Retrieval Rounds per Case**: Demonstrating efficient convergence (target: $1.2$ to $1.5$ rounds).
- **Fallback Resolution Rate**: Percentage of initial evidence gaps successfully resolved by the Round 2 targeted query.

### Medical Faithfulness & Safety Metrics
- **Unsupported Claim Rate**: Percentage of clinical statements in generated reports lacking direct support in retrieved chunks (evaluated via automated NLI / fact-checker; target: $0.0\%$).
- **Prediction Preservation**: Verification that ML classifier predictions are modified in $0.0\%$ of cases.

*Note on Ground-Truth Benchmark*: A formal test set of 50 paired clinical lesion profiles with expert-annotated ground-truth reference chunks must be established in `tests/data/rag_benchmark.json` to compute statistical metrics reproducibly.

The metric definitions and target values in this historical design are
proposals, not measured results. The quantitative Agentic RAG benchmark is
pending.

---

## 14. Current Architecture Status

The Agentic RAG proposal has been implemented:

- **Coordinator**: `app/rag/agentic/clinical_agent.py` (`ClinicalRAGAgent`).
- **Retrieval bound**: One or two rounds; configuration above two is rejected.
- **Query strategy**: Targeted retrieval queries derived from the ViT result and
  structured ABCDE/evolution features.
- **Evidence evaluation**: Sufficiency and missing-domain checks drive a
  targeted second round when needed.
- **Provenance**: The result records `[OBSERVED]`, `[ML MODEL]`, `[RETRIEVED]`,
  and `[UNAVAILABLE]` evidence plus a per-round trace.
- **Classifier authority**: Agentic RAG does not replace or modify the ViT
  classification result.
- **Fallback**: The application preserves conventional-RAG fallback metadata
  if Agentic RAG fails.
- **Benchmark status**: Quantitative Agentic RAG benchmarking remains pending;
  no metric values in this document represent measured results.
