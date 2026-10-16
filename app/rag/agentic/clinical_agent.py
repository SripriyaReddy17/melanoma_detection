"""
Clinical RAG Agent for Melanoma Analysis.

Implements an autonomous, bounded evidence retrieval and evaluation loop
grounded strictly in observed image features, ViT predictions, and clinical guidelines.

Reuses the existing retriever implementation in app.rag.retrieval.retrieve_documents.
Does NOT modify model predictions or invent missing clinical facts.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Callable

from app.rag.retrieval import retrieve_documents
from app.rag.agentic.evaluator import EvidenceEvaluator, DEFAULT_RELEVANCE_THRESHOLD

# Configure logger
logger = logging.getLogger("app.rag.agentic.clinical_agent")

# Bounded configuration constants
# NOTE: RELEVANCE_THRESHOLD is an engineering configuration, not a medical fact.
MAX_RETRIEVAL_ROUNDS: int = 2
RELEVANCE_THRESHOLD: float = DEFAULT_RELEVANCE_THRESHOLD
DEFAULT_TOP_K: int = 3


class ClinicalRAGAgent:
    """
    Autonomous clinical evidence retriever and synthesizer.

    Performs a bounded multi-round retrieval loop (max 2 rounds) to
    acquire grounded clinical literature for observed lesion findings.
    """

    def __init__(
        self,
        max_rounds: int = MAX_RETRIEVAL_ROUNDS,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        retriever_fn: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
    ):
        """
        Initialize the Clinical RAG Agent.

        Args:
            max_rounds: Maximum retrieval iterations (strictly bounded, default: 2).
            relevance_threshold: Engineering cosine similarity cutoff (default: 0.45).
            top_k: Number of documents to fetch per targeted query (default: 3).
            retriever_fn: Retrieval function to call; defaults to existing retrieve_documents.
        """
        if not 1 <= int(max_rounds) <= MAX_RETRIEVAL_ROUNDS:
            raise ValueError(
                f"max_rounds must be between 1 and {MAX_RETRIEVAL_ROUNDS}."
            )
        self.max_rounds = int(max_rounds)
        self.relevance_threshold = float(relevance_threshold)
        self.top_k = int(top_k)
        self.retriever_fn = retriever_fn or retrieve_documents
        self.evaluator = EvidenceEvaluator(relevance_threshold=self.relevance_threshold)

    # ========================================================
    # QUERY PLANNING
    # ========================================================

    def plan_initial_queries(
        self,
        classification: Dict[str, Any],
        abcde: Dict[str, Any],
        evolution: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Generate targeted clinical queries based strictly on observed system outputs.
        Does NOT invent missing clinical data.
        """
        queries: List[Dict[str, str]] = []

        # 1. Classification & Diagnostic Domain
        pred_class = str(classification.get("predicted_class", "unknown")).lower()
        mel_prob = float(classification.get("melanoma_probability", 0.0) or 0.0)

        if mel_prob >= 0.50 or pred_class in ["mel", "melanoma"]:
            queries.append({
                "domain": "classification",
                "query": (
                    "Cutaneous malignant melanoma clinical presentation, "
                    "radial vs vertical growth phase, and surgical excision guidelines."
                ),
            })
        else:
            queries.append({
                "domain": "classification",
                "query": (
                    "Benign melanocytic nevi dermoscopic stability, "
                    "structural symmetry, and diagnostic monitoring criteria."
                ),
            })

        # 2. Asymmetry (A)
        asym_val = abcde.get("asymmetry")
        if isinstance(asym_val, dict):
            asym_val = asym_val.get("score")
        if asym_val is not None:
            try:
                asym_score = float(asym_val)
                if asym_score >= 0.35:
                    queries.append({
                        "domain": "asymmetry",
                        "query": (
                            f"Dermoscopic asymmetry along orthogonal axes and differential "
                            f"melanocytic pigment distribution."
                        ),
                    })
            except (ValueError, TypeError):
                pass

        # 3. Border (B)
        border_val = abcde.get("border_irregularity")
        if isinstance(border_val, dict):
            border_val = border_val.get("border_irregularity_score")
        if border_val is not None:
            try:
                border_score = float(border_val)
                if border_score >= 0.35:
                    queries.append({
                        "domain": "border",
                        "query": (
                            f"Irregular notched borders, margin abrupt termination, "
                            f"and peripheral contour roughness in dermoscopy."
                        ),
                    })
            except (ValueError, TypeError):
                pass

        # 4. Color (C)
        color_val = abcde.get("color_index")
        if color_val is None and isinstance(abcde.get("color"), dict):
            color_val = abcde.get("color", {}).get("index")
        if color_val is not None:
            try:
                color_index = float(color_val)
                if color_index >= 0.30:
                    queries.append({
                        "domain": "color",
                        "query": (
                            f"Color variegation, multi-chromatic pigment distribution, "
                            f"and blue-gray or dark brown melanoma areas."
                        ),
                    })
            except (ValueError, TypeError):
                pass

        # 5. Diameter (D) - Only query mm thresholds if valid calibration exists
        dia_dict = abcde.get("diameter") if isinstance(abcde.get("diameter"), dict) else {}
        cal_used = bool(
            dia_dict.get("calibration_used")
            or abcde.get("diameter_mm_available")
            or abcde.get("calibration_used", False)
        )
        d_mm = dia_dict.get("max_diameter_mm") or abcde.get("diameter_mm")

        if cal_used and d_mm is not None:
            try:
                d_mm_val = float(d_mm)
                if d_mm_val >= 6.0:
                    queries.append({
                        "domain": "diameter",
                        "query": (
                            f"Lesion diameter exceeding 6mm ABCDE guideline and "
                            f"physical calibration requirements."
                        ),
                    })
                else:
                    queries.append({
                        "domain": "diameter",
                        "query": (
                            f"Lesion diameter assessment within standard reference "
                            f"thresholds in dermoscopy."
                        ),
                    })
            except (ValueError, TypeError):
                pass
        # If not calibrated, do NOT query mm thresholds; marked as UNAVAILABLE in provenance.

        # 6. Evolution (E) - Only query if longitudinal data is available
        evo_avail = bool(evolution and evolution.get("available", False))
        if evo_avail:
            chg = bool(evolution.get("change_detected", False))
            queries.append({
                "domain": "evolution",
                "query": (
                    f"Longitudinal lesion evolution and temporal change detection "
                    f"across sequential dermoscopic visits."
                ),
            })
        # If evolution is unavailable, do NOT pretend it exists; marked as UNAVAILABLE.

        return queries

    def plan_gap_queries(self, missing_domains: List[str], mel_prob: float) -> List[Dict[str, str]]:
        """
        Generate refined, focused gap queries for domains that lacked
        sufficient clinical evidence in Round 1.
        """
        gap_queries: List[Dict[str, str]] = []

        gap_templates = {
            "classification": (
                "Dermatology guidelines cutaneous melanoma suspicious lesion pathology"
                if mel_prob >= 0.50
                else "Benign melanocytic nevi dermoscopic criteria symmetry stability monitoring guidelines"
            ),
            "asymmetry": "ABCD rule asymmetry criteria melanocytic lesion contour reflection mismatch",
            "border": "Dermoscopy border notch structural irregularity margin contour",
            "color": "Dermoscopy color variegation shades pigment patterns melanocytic atypical nevi",
            "diameter": "Lesion diameter measurement calibration standard skin dermoscopy ABCD rule",
            "evolution": "Longitudinal dermoscopy lesion change monitoring criteria sequential imaging",
        }

        for domain in missing_domains:
            query_text = gap_templates.get(
                domain,
                f"Clinical dermoscopy evidence and guidelines for {domain} lesion assessment",
            )
            gap_queries.append({
                "domain": domain,
                "query": query_text,
            })

        return gap_queries

    # ========================================================
    # EVIDENCE PROVENANCE BUILDER
    # ========================================================

    def build_provenance(
        self,
        classification: Dict[str, Any],
        abcde: Dict[str, Any],
        evolution: Dict[str, Any],
        retrieved_documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build structured 4-tier provenance records:
        [OBSERVED], [ML MODEL], [RETRIEVED], [UNAVAILABLE].
        """
        provenance: List[Dict[str, Any]] = []

        # [ML MODEL] Findings
        pred = classification.get("predicted_class", "unknown")
        prob = float(classification.get("melanoma_probability", 0.0) or 0.0)
        risk = classification.get("risk_level", "unspecified")
        provenance.append({
            "category": "[ML MODEL]",
            "source": "Vision Transformer Classifier",
            "feature": "classification",
            "statement": (
                f"Predicted class: '{pred}' with estimated melanoma probability {prob:.1%} "
                f"(risk category: {risk})."
            ),
        })

        # [OBSERVED] Findings
        asym = abcde.get("asymmetry")
        if isinstance(asym, dict):
            asym = asym.get("score")
        if asym is not None:
            provenance.append({
                "category": "[OBSERVED]",
                "source": "Lesion Contour Analysis Pipeline",
                "feature": "asymmetry",
                "statement": f"Normalized asymmetry score: {float(asym):.2f}.",
            })

        border = abcde.get("border_irregularity")
        if isinstance(border, dict):
            border = border.get("border_irregularity_score")
        if border is not None:
            provenance.append({
                "category": "[OBSERVED]",
                "source": "Boundary Analysis Pipeline",
                "feature": "border",
                "statement": f"Normalized border irregularity score: {float(border):.2f}.",
            })

        color = abcde.get("color_index")
        if color is None and isinstance(abcde.get("color"), dict):
            color = abcde.get("color", {}).get("index")
        if color is not None:
            provenance.append({
                "category": "[OBSERVED]",
                "source": "HSV Color Analysis Pipeline",
                "feature": "color",
                "statement": f"Normalized color variation index: {float(color):.2f}.",
            })

        # Diameter [OBSERVED] vs [UNAVAILABLE]
        dia_dict = abcde.get("diameter") if isinstance(abcde.get("diameter"), dict) else {}
        cal_used = bool(
            dia_dict.get("calibration_used")
            or abcde.get("diameter_mm_available")
            or abcde.get("calibration_used", False)
        )
        d_mm = dia_dict.get("max_diameter_mm") or abcde.get("diameter_mm")
        d_px = dia_dict.get("max_diameter_pixels") or abcde.get("diameter_pixels")

        if cal_used and d_mm is not None:
            provenance.append({
                "category": "[OBSERVED]",
                "source": "Calibrated Dimensional Measurement",
                "feature": "diameter",
                "statement": f"Measured physical diameter: {float(d_mm):.1f} mm via verified calibration.",
            })
        else:
            if d_px is not None:
                provenance.append({
                    "category": "[OBSERVED]",
                    "source": "Pixel Extent Analysis",
                    "feature": "diameter_pixels",
                    "statement": f"Uncalibrated extent: approximately {float(d_px):.0f} pixels.",
                })
            provenance.append({
                "category": "[UNAVAILABLE]",
                "source": "Calibration Pipeline",
                "feature": "diameter_mm",
                "statement": (
                    "Physical diameter measurement was not available because no valid "
                    "calibration marker or ruler was detected."
                ),
            })

        # Evolution [OBSERVED] vs [UNAVAILABLE]
        evo_avail = bool(evolution and evolution.get("available", False))
        if evo_avail:
            chg = bool(evolution.get("change_detected", False))
            provenance.append({
                "category": "[OBSERVED]",
                "source": "Longitudinal Comparison Pipeline",
                "feature": "evolution",
                "statement": (
                    f"Sequential image comparison: "
                    f"{'significant change detected' if chg else 'no significant change detected'}."
                ),
            })
        else:
            provenance.append({
                "category": "[UNAVAILABLE]",
                "source": "Clinical History Input",
                "feature": "evolution",
                "statement": (
                    "Evolution history was not provided; longitudinal change cannot be "
                    "inferred from a single image."
                ),
            })

        # [RETRIEVED] Literature Findings
        for doc in retrieved_documents:
            source = doc.get("source", "Medical Reference")
            score = float(doc.get("score", 0.0) or 0.0)
            text_snippet = str(doc.get("text", "")).strip()
            provenance.append({
                "category": "[RETRIEVED]",
                "source": source,
                "feature": doc.get("domain", "medical_literature"),
                "score": round(score, 4),
                "statement": text_snippet,
            })

        return provenance

    # ========================================================
    # BOUNDED RETRIEVAL LOOP
    # ========================================================

    def run(
        self,
        classification: Dict[str, Any],
        abcde: Dict[str, Any],
        evolution: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the bounded Agentic RAG loop.

        Does NOT modify model predictions, probability, or ABCDE values.

        Returns:
            Backward-compatible dictionary containing:
                - query (str)
                - documents (List[Dict])
                - agent_trace (Dict)
                - provenance (List[Dict])
                - evidence_summary (str)
                - rounds_executed (int)
                - sufficiency (str)
                - is_agentic (bool)
        """
        classification = classification or {}
        abcde = abcde or {}
        evolution = evolution or {}
        k = int(top_k or self.top_k)

        logger.info(
            "ClinicalRAGAgent started: max_rounds=%d, relevance_threshold=%.2f, top_k=%d",
            self.max_rounds,
            self.relevance_threshold,
            k,
        )

        trace_rounds: List[Dict[str, Any]] = []
        all_retrieved_docs: List[Dict[str, Any]] = []
        executed_query_strings: List[str] = []

        # ====================================================
        # ROUND 1: Target Formulation & Retrieval
        # ====================================================
        round_1_queries = self.plan_initial_queries(
            classification=classification,
            abcde=abcde,
            evolution=evolution,
        )
        target_domains = [q["domain"] for q in round_1_queries]
        logger.info("Executing retrieval round 1/%d", self.max_rounds)
        logger.info("Round 1 query domains: %s", target_domains)

        round_1_docs: List[Dict[str, Any]] = []
        round_1_query_texts: List[str] = []

        for q_item in round_1_queries:
            q_text = q_item["query"]
            round_1_query_texts.append(q_text)
            executed_query_strings.append(q_text)

            try:
                retrieved = self.retriever_fn(q_text, k)
            except Exception as exc:
                logger.warning("Retrieval failed for query '%s': %s", q_text, exc)
                retrieved = []

            for doc in retrieved:
                round_1_docs.append(doc)

        # Deduplicate round 1 docs while keeping highest score
        round_1_docs_dedup = self._deduplicate_docs(round_1_docs)
        all_retrieved_docs.extend(round_1_docs_dedup)

        logger.info("Round 1 retrieved %d unique documents", len(round_1_docs_dedup))

        # Evaluate Round 1 evidence
        evaluation_r1 = self.evaluator.evaluate_evidence(
            retrieved_documents=round_1_docs_dedup,
            target_domains=target_domains,
        )

        logger.info(
            "Round 1 evidence sufficiency: %s (missing domains: %s)",
            "sufficient" if evaluation_r1["is_sufficient"] else "insufficient",
            evaluation_r1["missing_domains"],
        )

        should_retry = self.evaluator.should_retrieve_again(
            evaluation=evaluation_r1,
            current_round=1,
            max_rounds=self.max_rounds,
        )

        decision_r1 = "retrieve_again" if should_retry else "stop"

        trace_rounds.append({
            "iteration": 1,
            "queries": round_1_query_texts,
            "retrieved_documents": [
                {
                    "source": d.get("source", "Unknown"),
                    "chunk_index": d.get("chunk_index"),
                    "score": d.get("score", 0.0),
                }
                for d in round_1_docs_dedup
            ],
            "similarity_scores": evaluation_r1["similarity_scores"],
            "missing_domains": evaluation_r1["missing_domains"],
            "decision": decision_r1,
        })

        # ====================================================
        # ROUND 2: Focused Gap Retrieval (if needed & bounded)
        # ====================================================
        if should_retry and self.max_rounds >= 2:
            logger.info(
                "Triggering second retrieval round for missing domains: %s",
                evaluation_r1["missing_domains"],
            )
            mel_prob = float(classification.get("melanoma_probability", 0.0) or 0.0)
            gap_queries = self.plan_gap_queries(
                missing_domains=evaluation_r1["missing_domains"],
                mel_prob=mel_prob,
            )

            round_2_docs: List[Dict[str, Any]] = []
            round_2_query_texts: List[str] = []

            for g_item in gap_queries:
                g_text = g_item["query"]
                round_2_query_texts.append(g_text)
                executed_query_strings.append(g_text)

                try:
                    retrieved = self.retriever_fn(g_text, k)
                except Exception as exc:
                    logger.warning("Retrieval failed for gap query '%s': %s", g_text, exc)
                    retrieved = []

                for doc in retrieved:
                    round_2_docs.append(doc)

            round_2_docs_dedup = self._deduplicate_docs(round_2_docs)
            all_retrieved_docs.extend(round_2_docs_dedup)

            logger.info("Round 2 retrieved %d unique documents", len(round_2_docs_dedup))

            # Re-evaluate all accumulated evidence against original target domains
            all_dedup = self._deduplicate_docs(all_retrieved_docs)
            evaluation_r2 = self.evaluator.evaluate_evidence(
                retrieved_documents=all_dedup,
                target_domains=target_domains,
            )

            logger.info(
                "Round 2 evidence sufficiency: %s (missing domains: %s)",
                "sufficient" if evaluation_r2["is_sufficient"] else "insufficient",
                evaluation_r2["missing_domains"],
            )

            trace_rounds.append({
                "iteration": 2,
                "queries": round_2_query_texts,
                "retrieved_documents": [
                    {
                        "source": d.get("source", "Unknown"),
                        "chunk_index": d.get("chunk_index"),
                        "score": d.get("score", 0.0),
                    }
                    for d in round_2_docs_dedup
                ],
                "similarity_scores": [
                    float(d.get("score", 0.0) or 0.0)
                    for d in round_2_docs_dedup
                ],
                "missing_domains": evaluation_r2["missing_domains"],
                "decision": "stop",
            })
            final_evaluation = evaluation_r2
        else:
            final_evaluation = evaluation_r1

        # Final deduplication and ranking of retained documents
        final_docs = self._deduplicate_docs(all_retrieved_docs)
        final_docs.sort(key=lambda d: float(d.get("score", 0.0) or 0.0), reverse=True)
        top_final_docs = final_docs[:k]

        logger.info(
            "ClinicalRAGAgent finished: %d total evidence documents retained across %d rounds",
            len(top_final_docs),
            len(trace_rounds),
        )

        # Assemble structured agent trace
        agent_trace = {
            "agent_type": "clinical_rag_agent",
            "max_rounds": self.max_rounds,
            "rounds": trace_rounds,
        }

        # Build comprehensive provenance list
        provenance = self.build_provenance(
            classification=classification,
            abcde=abcde,
            evolution=evolution,
            retrieved_documents=top_final_docs,
        )

        # Build readable summary string
        summary_lines = []
        for p in provenance:
            if p["category"] in ["[OBSERVED]", "[ML MODEL]", "[UNAVAILABLE]"]:
                summary_lines.append(f"{p['category']} {p['statement']}")
        summary_text = "\n".join(summary_lines)

        primary_query = (
            round_1_query_texts[0]
            if round_1_query_texts
            else "Melanoma clinical guidelines and ABCDE assessment"
        )

        return {
            "query": primary_query,
            "documents": top_final_docs,
            "agent_trace": agent_trace,
            "provenance": provenance,
            "evidence_summary": summary_text,
            "rounds_executed": len(trace_rounds),
            "sufficiency": "sufficient" if final_evaluation["is_sufficient"] else "insufficient",
            "is_agentic": True,
        }

    # ========================================================
    # DOCUMENT DEDUPLICATION HELPER
    # ========================================================

    @staticmethod
    def _deduplicate_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate document list while preserving highest score."""
        seen: Dict[tuple, Dict[str, Any]] = {}
        for d in docs:
            if not d:
                continue
            key = (
                str(d.get("source")),
                d.get("chunk_index"),
                str(d.get("text", ""))[:80],
            )
            score = float(d.get("score", 0.0) or 0.0)
            if key not in seen or score > float(seen[key].get("score", 0.0) or 0.0):
                seen[key] = d
        return list(seen.values())
