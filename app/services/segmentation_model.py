"""Segmentation adapter — UltraLight VM-UNet only.

All clinical features (A/B/C/D/E) are derived from a mask
produced by UltraLight VM-UNet. Segmentation failures propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.segmentation import SEGMENTATION_WEIGHTS, segment_lesion


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class SegmentationResult:
    """Result returned by the VM-UNet adapter."""
    model: str
    status: str          # "ready" after successful VM-UNet inference
    quality_score: float  # foreground pixel fraction (0-1)
    note: str
    mask: np.ndarray | None = None  # binary uint8 mask, values 0/1


# ============================================================
# ULTRALIGHT VM-UNET ADAPTER
# ============================================================

class UltraLightVMUNetAdapter:
    """
    Adapter for UltraLight VM-UNet dermoscopic segmentation.

    Uses the pre-trained checkpoint at
        models/segmentation/UltraLight_VM_UNet.pth
    (configured in app.services.segmentation).
    """

    MODEL_NAME = "UltraLight VM-UNet"

    def __init__(self) -> None:
        self.is_available = SEGMENTATION_WEIGHTS.is_file()

    def segment(
        self,
        image_path: str,
        remove_hair_flag: bool = False,
        exclude_polygon=None,
    ) -> SegmentationResult:
        """
        Run lesion segmentation on *image_path*.

        Returns a :class:`SegmentationResult` after VM-UNet produces
        a binary mask. Raises when the model cannot produce a mask.
        """
        mask = segment_lesion(
            image_path,
            remove_hair_flag=remove_hair_flag,
            exclude_polygon=exclude_polygon,
        )
        quality = float((mask > 0).mean())
        quality = max(0.0, min(1.0, quality))

        return SegmentationResult(
            model=self.MODEL_NAME,
            status="ready",
            quality_score=quality,
            note=(
                "UltraLight VM-UNet segmentation completed. "
                "All ABCDE clinical features are derived from "
                "this mask."
            ),
            mask=mask,
        )


# ============================================================
# MODULE-LEVEL ADAPTER SINGLETON
# ============================================================

segmentation_adapter = UltraLightVMUNetAdapter()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def segment_with_vmunet(
    image_path: str,
    remove_hair_flag: bool = False,
    exclude_polygon=None,
) -> dict[str, Any]:
    """
    Run VM-UNet segmentation and return a structured result dict.

    The returned dict is consumed by the multimodal pipeline and
    stored in MongoDB via the analysis record.

    Keys
    ----
    model           : str   — model name
    status          : str   — "ready" after successful VM-UNet inference
    quality_score   : float — foreground pixel fraction
    note            : str   — human-readable status note
    mask_available  : bool  — True when a mask array was produced
    """
    if not image_path:
        return {
            "model": UltraLightVMUNetAdapter.MODEL_NAME,
            "status": "not_run",
            "quality_score": 0.0,
            "note": "No image path supplied; segmentation not executed.",
            "mask_available": False,
        }

    result = UltraLightVMUNetAdapter().segment(
        image_path,
        remove_hair_flag=remove_hair_flag,
        exclude_polygon=exclude_polygon,
    )

    return {
        "model": result.model,
        "status": result.status,
        "quality_score": result.quality_score,
        "note": result.note,
        "mask_available": result.mask is not None,
    }
