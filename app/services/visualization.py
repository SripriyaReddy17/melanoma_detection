"""
XAI visualization overlays generated from the image + lesion mask.

Each function returns PNG/JPEG bytes ready to be served by the API:
- raw          : original photo
- cleaned      : hair-removed (DullRazor-style) photo
- mask         : binary lesion mask
- mask_overlay : mask overlaid on photo
- asymmetry    : principal axes drawn on the lesion
- border       : contour curvature visualization
- caliper      : diameter measurement line (like a dermatoscope caliper)
"""

import io

import cv2
import numpy as np

from app.services.segmentation import segment_lesion, remove_hair


def _encode(image_bgr: np.ndarray, fmt: str = ".png") -> bytes:
    ok, buf = cv2.imencode(fmt, image_bgr)
    if not ok:
        raise ValueError("Could not encode visualization image.")
    return buf.tobytes()


def _load(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("Could not load image.")
    return img


def _lesion_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def farthest_points(mask: np.ndarray):
    """Two farthest points on the lesion convex hull (= clinical diameter)."""
    contour = _lesion_contour(mask)
    if contour is None or len(contour) < 5:
        return None, 0.0
    hull = cv2.convexHull(contour).reshape(-1, 2)
    max_d, best = 0.0, (hull[0], hull[-1])
    for i in range(len(hull)):
        d = np.linalg.norm(hull - hull[i], axis=1)
        j = int(np.argmax(d))
        if d[j] > max_d:
            max_d = float(d[j])
            best = (tuple(hull[i]), tuple(hull[j]))
    return best, max_d


def vis_raw(image_path: str, remove_hair_flag: bool = False) -> bytes:
    img = _load(image_path)
    if remove_hair_flag:
        img = remove_hair(img)
    return _encode(img, ".jpg")


def vis_mask(image_path: str, remove_hair_flag: bool = False) -> bytes:
    mask = segment_lesion(image_path, remove_hair_flag=remove_hair_flag)
    out = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return _encode(out)


def vis_mask_overlay(image_path: str, remove_hair_flag: bool = False) -> bytes:
    img = _load(image_path)
    if remove_hair_flag:
        img = remove_hair(img)
    mask = segment_lesion(image_path, remove_hair_flag=remove_hair_flag)
    overlay = img.copy()
    overlay[mask == 1] = (0.35 * overlay[mask == 1] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    contour = _lesion_contour(mask)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
    return _encode(overlay, ".jpg")


def vis_asymmetry(image_path: str, remove_hair_flag: bool = False) -> bytes:
    """Draw principal (major/minor) axes used for the asymmetry score."""
    img = _load(image_path)
    if remove_hair_flag:
        img = remove_hair(img)
    mask = segment_lesion(image_path, remove_hair_flag=remove_hair_flag)

    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return _encode(img, ".jpg")

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)[:2]
    cx, cy = int(mean[0][0]), int(mean[0][1])

    projections = (pts - mean) @ eigenvectors.T
    t_max = float(np.abs(projections[:, 0]).max())
    n_max = float(np.abs(projections[:, 1]).max())

    v_major = eigenvectors[0] * t_max
    v_minor = eigenvectors[1] * n_max

    cv2.line(img,
             (int(cx - v_major[0]), int(cy - v_major[1])),
             (int(cx + v_major[0]), int(cy + v_major[1])), (255, 80, 80), 3)
    cv2.line(img,
             (int(cx - v_minor[0]), int(cy - v_minor[1])),
             (int(cx + v_minor[0]), int(cy + v_minor[1])), (80, 200, 255), 3)
    cv2.circle(img, (cx, cy), 6, (255, 255, 255), -1)

    cv2.putText(img, "major axis", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 80), 2)
    cv2.putText(img, "minor axis", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 255), 2)
    return _encode(img, ".jpg")


def vis_border(image_path: str, remove_hair_flag: bool = False) -> bytes:
    """Color the lesion contour by local curvature (red = irregular)."""
    img = _load(image_path)
    if remove_hair_flag:
        img = remove_hair(img)
    mask = segment_lesion(image_path, remove_hair_flag=remove_hair_flag)
    contour = _lesion_contour(mask)
    if contour is None or len(contour) < 15:
        return _encode(img, ".jpg")

    pts = contour.reshape(-1, 2)
    curvatures = np.zeros(len(pts))
    for i in range(len(pts)):
        p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        v1, v2 = p2 - p1, p3 - p2
        a1, a2 = np.arctan2(v1[1], v1[0]), np.arctan2(v2[1], v2[0])
        delta = abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
        ds = np.linalg.norm(v2)
        curvatures[i] = delta / ds if ds > 1e-6 else 0

    cmax = curvatures.max() or 1.0
    for p, c in zip(pts, curvatures):
        t = min(c / cmax, 1.0)
        color = (0, int(255 * (1 - t)), 255)          # yellow -> red
        cv2.circle(img, tuple(p), 2, color, -1)
    return _encode(img, ".jpg")


def vis_caliper(image_path: str, scale_px_per_mm: float | None = None,
                remove_hair_flag: bool = False) -> bytes:
    """Dermatoscope-style caliper line across the widest lesion extent."""
    img = _load(image_path)
    if remove_hair_flag:
        img = remove_hair(img)
    mask = segment_lesion(image_path, remove_hair_flag=remove_hair_flag)

    (p1, p2), dist = farthest_points(mask)
    if p1 is None:
        return _encode(img, ".jpg")

    cv2.line(img, p1, p2, (0, 255, 120), 3)
    for p in (p1, p2):
        cv2.circle(img, p, 7, (0, 255, 120), -1)
        cv2.circle(img, p, 9, (255, 255, 255), 2)

    label = f"{dist:.0f} px"
    if scale_px_per_mm and scale_px_per_mm > 0:
        label += f"  ({dist / scale_px_per_mm:.1f} mm)"

    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (mid[0] - tw // 2 - 8, mid[1] - th - 14),
                  (mid[0] + tw // 2 + 8, mid[1] + 8), (20, 20, 30), -1)
    cv2.putText(img, label, (mid[0] - tw // 2, mid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)
    return _encode(img, ".jpg")


VIEWS = {
    "raw": lambda p, s, h: vis_raw(p, h),
    "cleaned": lambda p, s, h: vis_raw(p, True),
    "mask": lambda p, s, h: vis_mask(p, h),
    "overlay": lambda p, s, h: vis_mask_overlay(p, h),
    "asymmetry": lambda p, s, h: vis_asymmetry(p, h),
    "border": lambda p, s, h: vis_border(p, h),
    "caliper": lambda p, s, h: vis_caliper(p, s, h),
}


def render_view(view: str, image_path: str,
                scale_px_per_mm: float | None = None,
                remove_hair_flag: bool = False) -> bytes:
    fn = VIEWS.get(view)
    if not fn:
        raise ValueError(f"Unknown view '{view}'.")
    return fn(image_path, scale_px_per_mm, remove_hair_flag)
