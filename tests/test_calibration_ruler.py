import cv2
import numpy as np

from app.services.calibration import detect_calibration_reference


def synthetic_ruler():
    image = np.full((600, 900, 3), 205, dtype=np.uint8)

    # 1 mm ruler with 10 px / mm and a 70 mm baseline.
    start = np.array([170.0, 500.0])
    direction = np.array([0.92, -0.39])
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])

    length_mm = 70
    px_per_mm = 10.0
    end = start + direction * (length_mm * px_per_mm)

    cv2.line(
        image,
        tuple(np.round(start).astype(int)),
        tuple(np.round(end).astype(int)),
        (25, 25, 25),
        3,
    )

    for mm in range(length_mm + 1):
        center = start + direction * (mm * px_per_mm)
        tick_len = 22 if mm % 10 == 0 else 14
        p1 = center - normal * tick_len / 2
        p2 = center + normal * tick_len / 2
        cv2.line(
            image,
            tuple(np.round(p1).astype(int)),
            tuple(np.round(p2).astype(int)),
            (25, 25, 25),
            2,
        )

    return image


def test_ruler_calibration():
    image = synthetic_ruler()
    result = detect_calibration_reference(image)
    assert result["valid"]
    assert result["reference_type"] == "ruler_1mm"
    assert result["pixels_per_mm"] > 0
    assert abs(result["pixels_per_mm"] - 10.0) < 2.0
