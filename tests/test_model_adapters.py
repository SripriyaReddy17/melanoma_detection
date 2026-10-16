from app.services.clip_model import CLIPConceptExtractor
from app.services.llm_report import DeepSeekReportSynthesizer
import pytest

from app.services.segmentation import SegmentationError
from app.services.segmentation_model import UltraLightVMUNetAdapter
from app.services.vision_model import VisionTransformerAdapter


def test_vision_model_adapter_reports_backbone_and_status():
    result = VisionTransformerAdapter().predict()
    assert result.backbone == "google/vit-base-patch16-224"
    assert result.status in {"available", "fallback"}


def test_segmentation_adapter_propagates_segmentation_failure(monkeypatch):
    def fail_segmentation(*args, **kwargs):
        raise SegmentationError("VM-UNet unavailable")

    monkeypatch.setattr(
        "app.services.segmentation_model.segment_lesion",
        fail_segmentation,
    )

    with pytest.raises(SegmentationError):
        UltraLightVMUNetAdapter().segment("dummy.png")


def test_clip_and_llm_adapters_are_configurable():
    clip = CLIPConceptExtractor().extract(
        image_features={"texture": 0.8},
        clinical_features={"border_irregularity": 0.7, "asymmetry": 0.6, "color": {"index": 0.8}},
    )
    llm = DeepSeekReportSynthesizer().generate("High risk lesion with asymmetric border.")

    assert "concepts" in clip
    assert llm["format"] == "medical_style_report"
    assert llm["status"] in {"ready", "fallback"}
