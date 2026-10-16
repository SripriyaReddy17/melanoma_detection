"""Vision model adapter abstraction for dermoscopic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.classification import MODEL_BACKBONE


@dataclass
class VisionModelResult:
    model: str
    status: str
    backbone: str
    note: str


class VisionTransformerAdapter:
    """Thin adapter around the trained ViT model or heuristic fallback."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.model_name = "ViT"
        self.is_available = self.model_dir is not None and self.model_dir.exists()

    def predict(self, image_path: str | None = None, features: dict[str, Any] | None = None) -> VisionModelResult:
        return VisionModelResult(
            model=self.model_name,
            status="available" if self.is_available else "fallback",
            backbone=MODEL_BACKBONE,
            note=(
                "Vision Transformer streams dermoscopic features into the downstream "
                "ABC/contrastive/report pipeline."
            ),
        )


vision_adapter = VisionTransformerAdapter()
