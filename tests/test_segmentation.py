import cv2
import numpy as np
import pytest
import torch

from app.services import segmentation


def test_missing_checkpoint_raises_segmentation_error(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(segmentation, "SEGMENTATION_WEIGHTS", tmp_path / "missing.pth")
    monkeypatch.setattr(segmentation, "_vmunet_model", None)

    with pytest.raises(segmentation.SegmentationError, match="checkpoint not found"):
        segmentation.segment_lesion(image_path)


def test_checkpoint_load_failure_raises_segmentation_error(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    checkpoint = tmp_path / "checkpoint.pth"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    checkpoint.touch()
    monkeypatch.setattr(segmentation, "SEGMENTATION_WEIGHTS", checkpoint)
    monkeypatch.setattr(segmentation, "_vmunet_model", None)

    def fail_load(*args, **kwargs):
        raise RuntimeError("checkpoint is unreadable")

    monkeypatch.setattr(segmentation.torch, "load", fail_load)

    with pytest.raises(segmentation.SegmentationError, match="Could not load"):
        segmentation.segment_lesion(image_path)


def test_vm_unet_inference_failure_does_not_fall_back(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))

    class BrokenModel(torch.nn.Module):
        def forward(self, inputs):
            raise RuntimeError("inference broke")

    monkeypatch.setattr(segmentation, "_get_vmunet_model", lambda: BrokenModel())

    with pytest.raises(segmentation.SegmentationError, match="inference failed"):
        segmentation.segment_lesion(image_path)


def test_production_segmentation_has_no_deep_model_bypass():
    with pytest.raises(TypeError, match="use_deep_model"):
        segmentation.segment_lesion("image.png", use_deep_model=False)
