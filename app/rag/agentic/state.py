"""
Analysis State Schema for Agentic RAG in CEFM (Rule 26).
Defines the structured context passed through all agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class AnalysisState:
    """Structured Analysis State containing all pipeline data."""
    image: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    abcde: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)
    contrastive: Dict[str, Any] = field(default_factory=dict)
    clip: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnalysisState:
        return cls(
            image=data.get("image", {}),
            validation=data.get("validation", {}),
            segmentation=data.get("segmentation", {}),
            abcde=data.get("abcde", {}),
            classification=data.get("classification", {}),
            contrastive=data.get("contrastive", {}),
            clip=data.get("clip", {}),
            retrieval=data.get("retrieval", {}),
            evidence=data.get("evidence", {}),
            report=data.get("report", {}),
            metadata=data.get("metadata", {}),
        )
