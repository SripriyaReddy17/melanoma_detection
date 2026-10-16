import cv2
import numpy as np
from app.services.calibration import detect_calibration_marker, CALIBRATION_MARKER_SIZE_MM
from app.services.image_validation import check_post_segmentation_quality


def synthetic(marker=True):
    img=np.full((600,800,3),190,dtype=np.uint8)
    cv2.ellipse(img,(390,300),(80,55),20,0,360,(55,45,40),-1)
    mask=np.zeros((600,800),dtype=np.uint8)
    cv2.ellipse(mask,(390,300),(80,55),20,0,360,255,-1)
    if marker:
        cv2.rectangle(img,(100,100),(180,180),(10,10,10),-1)
    return img,mask


def test_marker_calibration():
    img,mask=synthetic(True)
    result=detect_calibration_marker(img,mask)
    assert result["valid"]
    assert result["marker_size_mm"] == CALIBRATION_MARKER_SIZE_MM
    assert result["pixels_per_mm"] > 0


def test_missing_marker_rejected():
    img,mask=synthetic(False)
    result=detect_calibration_marker(img,mask)
    assert not result["valid"]


def test_post_quality_rejects_missing_marker():
    img,mask=synthetic(False)
    result=check_post_segmentation_quality(img,mask,{"valid":False})
    assert not result["valid"]


def test_boundary_rejection_message_is_segmentation_focused():
    img, mask = synthetic(False)
    mask[:, :3] = 1
    result = check_post_segmentation_quality(img, mask, {"valid": True, "pixels_per_mm": 5})
    assert not result["valid"]
    assert "segmentation" in result["reason"].lower() or "isolated" in result["reason"].lower()
    assert "lesion itself is cropped" not in result["reason"].lower()
