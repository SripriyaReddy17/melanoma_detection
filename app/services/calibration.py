"""Physical-scale calibration for lesion diameter estimation.

Supported references
--------------------
1. Standardized 20 mm x 20 mm dark square marker.
2. A straight ruler with clearly visible 1 mm graduations.

For patient images, the reference must be visible in the same image as the
lesion.  The detector returns a pixels-per-millimetre scale and a polygon that
can be excluded from lesion segmentation.
"""

from typing import Optional

import cv2
import numpy as np


CALIBRATION_MARKER_SIZE_MM = 20.0
RULER_TICK_SIZE_MM = 1.0

MIN_MARKER_SIDE_PX = 30.0
MAX_MARKER_AREA_FRACTION = 0.30
MAX_MARKER_LESION_DISTANCE_FRACTION = 0.65
MIN_MARKER_CONFIDENCE = 0.82

MIN_RULER_BASELINE_LENGTH_FRACTION = 0.18
MIN_RULER_TICKS = 5
MIN_RULER_CONFIDENCE = 0.72


def _empty_result(message: str = "No valid calibration reference detected.") -> dict:
    return {
        "valid": False,
        "reference_type": None,
        "pixels_per_mm": None,
        "marker_side_pixels": None,
        "marker_size_mm": CALIBRATION_MARKER_SIZE_MM,
        "tick_spacing_pixels": None,
        "tick_size_mm": RULER_TICK_SIZE_MM,
        "confidence": 0.0,
        "corners": None,
        "exclude_polygon": None,
        "message": message,
    }


def _lesion_box(mask: Optional[np.ndarray]):
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _quad_metrics(points: np.ndarray):
    pts = points.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0],
    )
    pts = pts[np.argsort(angles)]
    sides = np.linalg.norm(
        np.roll(pts, -1, axis=0) - pts,
        axis=1,
    )
    if np.min(sides) <= 0:
        return None
    ratio = float(np.max(sides) / np.min(sides))
    angles_deg = []
    for i in range(4):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % 4] - pts[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return None
        cosang = np.clip(
            float(np.dot(a, b) / denom),
            -1.0,
            1.0,
        )
        angles_deg.append(float(np.degrees(np.arccos(cosang))))
    side_mean = float(np.mean(sides))
    return pts, center, sides, ratio, angles_deg, side_mean


def _candidate_square_marker(
    image_rgb: np.ndarray,
    lesion_mask: Optional[np.ndarray],
) -> dict:
    """Detect the standardized 20 mm square marker."""
    empty = _empty_result("20 mm square calibration marker not detected.")

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = gray.shape
    image_area = float(h * w)
    lesion_box = _lesion_box(lesion_mask)
    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_MARKER_SIDE_PX ** 2 * 0.35:
            continue
        if area > MAX_MARKER_AREA_FRACTION * image_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.035 * perimeter,
            True,
        )

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        metrics = _quad_metrics(approx)
        if metrics is None:
            continue

        pts, center, sides, ratio, angles, side_mean = metrics
        if side_mean < MIN_MARKER_SIDE_PX:
            continue
        if ratio > 1.35:
            continue
        if min(angles) < 70 or max(angles) > 110:
            continue

        marker_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            marker_mask,
            np.round(pts).astype(np.int32),
            255,
        )

        mean_gray = float(cv2.mean(gray, mask=marker_mask)[0])
        if mean_gray > 95:
            continue

        if lesion_box is not None:
            x0, y0, x1, y1 = lesion_box
            bx0 = float(pts[:, 0].min())
            by0 = float(pts[:, 1].min())
            bx1 = float(pts[:, 0].max())
            by1 = float(pts[:, 1].max())
            if bx1 > x0 and bx0 < x1 and by1 > y0 and by0 < y1:
                continue

            lesion_center = np.array(
                [(x0 + x1) / 2, (y0 + y1) / 2],
                dtype=np.float32,
            )
            distance = float(np.linalg.norm(center - lesion_center))
            max_distance = MAX_MARKER_LESION_DISTANCE_FRACTION * np.hypot(w, h)
            if distance > max_distance:
                continue

        squareness = max(
            0.0,
            1.0 - abs(ratio - 1.0) / 0.35,
        )
        angle_error = max(abs(angle - 90) for angle in angles)
        angle_score = max(0.0, 1.0 - angle_error / 20.0)
        darkness = max(
            0.0,
            min(1.0, (130.0 - mean_gray) / 80.0),
        )
        confidence = (
            0.45 * squareness
            + 0.35 * angle_score
            + 0.20 * darkness
        )

        candidates.append(
            (confidence, side_mean, pts, center)
        )

    if not candidates:
        return empty

    confidence, side_mean, pts, center = max(
        candidates,
        key=lambda item: item[0],
    )

    if confidence < MIN_MARKER_CONFIDENCE:
        empty["message"] = (
            "A possible square reference was found, but it is not a reliable "
            "20 mm calibration marker."
        )
        empty["confidence"] = round(float(confidence), 3)
        return empty

    ppm = side_mean / CALIBRATION_MARKER_SIZE_MM
    if ppm <= 0:
        return empty

    return {
        "valid": True,
        "reference_type": "marker_20mm",
        "pixels_per_mm": round(float(ppm), 4),
        "marker_side_pixels": round(float(side_mean), 2),
        "marker_size_mm": CALIBRATION_MARKER_SIZE_MM,
        "tick_spacing_pixels": None,
        "tick_size_mm": None,
        "confidence": round(float(confidence), 3),
        "center_x": round(float(center[0]), 2),
        "center_y": round(float(center[1]), 2),
        "corners": np.round(pts, 2).tolist(),
        "exclude_polygon": np.round(pts, 2).tolist(),
        "message": (
            f"20 mm calibration marker detected ({side_mean:.1f} px)."
        ),
    }


def _line_points_from_hough(line):
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 0:
        return None
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return x1, y1, x2, y2, length, ux, uy, nx, ny, angle


def _detect_ruler_on_line(
    gray: np.ndarray,
    line_data: tuple,
) -> Optional[dict]:
    """Estimate 1 mm tick spacing along one long ruler baseline."""
    x1, y1, x2, y2, length, ux, uy, nx, ny, _ = line_data
    h, w = gray.shape

    half_width = int(max(10, min(24, 0.015 * min(h, w))))
    sample_count = int(max(300, min(1800, round(length * 1.5))))
    t = np.linspace(0.0, length, sample_count)

    xs = x1 + ux * t
    ys = y1 + uy * t

    scores = np.zeros(sample_count, dtype=np.float32)

    # Tick marks extend roughly perpendicular to the ruler baseline.
    # Count dark pixels away from the central baseline on both sides.
    for offset in range(3, half_width + 1):
        for sign in (-1.0, 1.0):
            px = np.rint(xs + nx * offset * sign).astype(np.int32)
            py = np.rint(ys + ny * offset * sign).astype(np.int32)
            valid = (
                (px >= 0) & (px < w) &
                (py >= 0) & (py < h)
            )
            if np.any(valid):
                values = gray[py[valid], px[valid]]
                scores[valid] += (values < 115).astype(np.float32)

    # Smooth the projection slightly.
    smooth_kernel = np.ones(5, dtype=np.float32) / 5.0
    scores = np.convolve(scores, smooth_kernel, mode="same")

    # Remove the baseline/DC component before periodicity analysis.
    periodic = scores.astype(np.float32) - float(np.median(scores))
    periodic[periodic < 0] = 0

    if float(np.max(periodic)) <= 0:
        return None

    # ------------------------------------------------------------
    # Estimate the fundamental tick spacing by autocorrelation.
    # This is more stable than relying on one arbitrary threshold.
    # ------------------------------------------------------------

    centered = periodic - float(np.mean(periodic))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-6:
        return None

    max_lag = min(140, sample_count // 3)
    min_lag = 4
    autocorr = np.zeros(max_lag + 1, dtype=np.float32)

    for lag in range(min_lag, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-6:
            continue
        autocorr[lag] = float(np.dot(a, b) / denom)

    # Look for a strong early local maximum. Tick spacing in ordinary
    # patient photographs should be neither extremely small nor enormous.
    local_candidates = []
    for lag in range(min_lag + 1, max_lag):
        value = autocorr[lag]
        if value <= 0:
            continue
        if value >= autocorr[lag - 1] and value >= autocorr[lag + 1]:
            local_candidates.append((float(value), lag))

    if not local_candidates:
        return None

    # Prefer an early fundamental period. A later harmonic (2x, 3x, ...)
    # is penalized so that a 1 mm ruler is not accidentally interpreted as
    # a 2 mm interval merely because its major ticks are stronger.
    best_value, best_lag = max(
        local_candidates,
        key=lambda item: item[0] * (1.0 / np.sqrt(item[1]))
    )

    if best_value < 0.18:
        return None

    # Convert lag in sampled coordinates back to pixels along the ruler.
    raw_spacing = float(best_lag) * length / max(sample_count - 1, 1)

    if raw_spacing < 4.0 or raw_spacing > 140.0:
        return None

    # ------------------------------------------------------------
    # Find individual tick centers using the detected period.
    # ------------------------------------------------------------

    local_threshold = max(
        float(np.percentile(scores, 70)),
        float(np.mean(scores) + 0.55 * np.std(scores)),
    )

    candidates = []
    radius = max(2, int(round(0.20 * best_lag)))

    for idx in range(radius, sample_count - radius):
        value = float(scores[idx])
        if value < local_threshold:
            continue
        if value != float(np.max(scores[idx - radius:idx + radius + 1])):
            continue
        candidates.append(idx)

    # Merge duplicates around the same physical tick.
    min_sep_samples = max(
        3,
        int(round(0.55 * best_lag)),
    )

    peaks = []
    for idx in candidates:
        if not peaks:
            peaks.append(idx)
            continue
        if idx - peaks[-1] >= min_sep_samples:
            peaks.append(idx)
        elif scores[idx] > scores[peaks[-1]]:
            peaks[-1] = idx

    if len(peaks) < MIN_RULER_TICKS:
        return None

    peak_positions = np.asarray(
        [t[idx] for idx in peaks],
        dtype=np.float32,
    )

    spacings = np.diff(peak_positions)
    spacings = spacings[spacings > 0]
    if len(spacings) < MIN_RULER_TICKS - 1:
        return None

    median_spacing = float(np.median(spacings))
    if median_spacing < 4.0 or median_spacing > 140.0:
        return None

    cv = float(
        np.std(spacings) / max(median_spacing, 1e-6)
    )

    close_fraction = float(
        np.mean(
            np.abs(spacings - median_spacing)
            <= 0.22 * median_spacing
        )
    )

    if cv > 0.32 or close_fraction < 0.60:
        return None

    # Require the measured spacing to agree reasonably with the periodicity
    # estimate. This protects against unrelated lines producing accidental
    # local maxima.
    period_error = abs(median_spacing - raw_spacing) / max(raw_spacing, 1e-6)
    if period_error > 0.35:
        return None

    # ------------------------------------------------------------
    # Build an exclusion strip around the ruler.
    # ------------------------------------------------------------

    start_t = float(peak_positions[0])
    end_t = float(peak_positions[-1])
    strip_half = max(14.0, 0.75 * median_spacing)

    p1 = np.array([
        x1 + ux * start_t + nx * strip_half,
        y1 + uy * start_t + ny * strip_half,
    ])
    p2 = np.array([
        x1 + ux * end_t + nx * strip_half,
        y1 + uy * end_t + ny * strip_half,
    ])
    p3 = np.array([
        x1 + ux * end_t - nx * strip_half,
        y1 + uy * end_t - ny * strip_half,
    ])
    p4 = np.array([
        x1 + ux * start_t - nx * strip_half,
        y1 + uy * start_t - ny * strip_half,
    ])

    count_score = min(1.0, len(peak_positions) / 10.0)
    regularity_score = max(0.0, 1.0 - min(cv, 0.32) / 0.32)
    periodic_score = max(0.0, min(1.0, best_value / 0.70))
    agreement_score = max(0.0, 1.0 - min(period_error, 0.35) / 0.35)

    confidence = (
        0.35 * regularity_score
        + 0.25 * count_score
        + 0.20 * periodic_score
        + 0.20 * agreement_score
    )

    if confidence < MIN_RULER_CONFIDENCE:
        return None

    return {
        "valid": True,
        "reference_type": "ruler_1mm",
        "pixels_per_mm": round(median_spacing / RULER_TICK_SIZE_MM, 4),
        "marker_side_pixels": None,
        "marker_size_mm": None,
        "tick_spacing_pixels": round(median_spacing, 3),
        "tick_size_mm": RULER_TICK_SIZE_MM,
        "confidence": round(float(confidence), 3),
        "ruler_angle_degrees": round(line_data[-1], 2),
        "tick_positions_pixels": [
            round(float(value), 2)
            for value in peak_positions
        ],
        "corners": None,
        "exclude_polygon": np.round(
            np.vstack([p1, p2, p3, p4]),
            2,
        ).tolist(),
        "message": (
            f"1 mm ruler detected ({median_spacing:.1f} px per mm, "
            f"{len(peak_positions)} graduations)."
        ),
    }

def _candidate_ruler(
    image_rgb: np.ndarray,
) -> dict:
    """Detect a straight ruler with 1 mm graduations."""
    empty = _empty_result(
        "No ruler with reliable 1 mm graduations was detected."
    )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(
        gray,
        50,
        140,
    )

    h, w = gray.shape
    min_line_length = max(
        80.0,
        MIN_RULER_BASELINE_LENGTH_FRACTION * min(h, w),
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=55,
        minLineLength=int(min_line_length),
        maxLineGap=20,
    )

    if lines is None:
        return empty

    candidates = []

    # OpenCV may return HoughLinesP output as either (N, 1, 4) or (N, 4).
    # Normalize it to a simple (N, 4) array before iterating.
    lines = np.asarray(lines).reshape(-1, 4)

    for line in lines:
        line_data = _line_points_from_hough(line)
        if line_data is None:
            continue

        x1, y1, x2, y2, length, ux, uy, nx, ny, angle = line_data

        # Extremely short lines are not rulers.
        if length < min_line_length:
            continue

        # Prefer long clean lines near the image perimeter, since the ruler
        # in our protocol is placed beside rather than over the lesion.
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        edge_distance = min(
            mx,
            my,
            w - mx,
            h - my,
        )
        edge_score = 1.0 - min(
            1.0,
            edge_distance / (0.50 * min(h, w)),
        )

        result = _detect_ruler_on_line(
            gray,
            line_data,
        )

        if result is None:
            continue

        score = (
            0.60 * result["confidence"]
            + 0.25 * min(1.0, length / max(min_line_length, 1.0) / 2.0)
            + 0.15 * edge_score
        )

        result["_selection_score"] = float(score)
        candidates.append(result)

    if not candidates:
        return empty

    best = max(
        candidates,
        key=lambda item: item["_selection_score"],
    )
    best.pop("_selection_score", None)
    return best


def detect_calibration_reference(
    image_rgb: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect either the standard square marker or a 1 mm ruler.

    The standardized 20 mm marker is preferred. If it is not found,
    the detector falls back to a ruler with visible 1 mm graduations.
    """
    if image_rgb is None or image_rgb.size == 0:
        return _empty_result(
            "Empty image; calibration cannot be performed."
        )

    marker = _candidate_square_marker(
        image_rgb,
        lesion_mask,
    )

    if marker.get("valid"):
        return marker

    ruler = _candidate_ruler(
        image_rgb,
    )

    if ruler.get("valid"):
        return ruler

    return _empty_result(
        "No valid 20 mm calibration marker or 1 mm graduated ruler was detected. "
        "Please retake the image with a complete calibration reference visible beside the lesion."
    )


# Backward-compatible names used by the existing pipeline.
def detect_calibration_marker(
    image_rgb: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
) -> dict:
    """Backward-compatible alias for the combined calibration detector."""
    return detect_calibration_reference(
        image_rgb,
        lesion_mask=lesion_mask,
    )


def estimate_pixels_per_mm(
    image_rgb: np.ndarray,
    lesion_mask: Optional[np.ndarray] = None,
    **_ignored,
) -> dict:
    """Return calibration information from a marker or 1 mm ruler."""
    return detect_calibration_reference(
        image_rgb,
        lesion_mask=lesion_mask,
    )
