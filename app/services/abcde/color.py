"""
Color Variation (C) feature extraction for CEFM.
Complies with Table 1 of CEFM Paper:
sigma_H, sigma_S, sigma_V (HSV standard deviations)
and Rule 14 descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np


# Maximum standard deviation for a variable bounded in [0, L] is L/2.
_H_STD_MAX = 179.0 / 2.0
_SV_STD_MAX = 255.0 / 2.0


def compute_color_variation(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Return comprehensive color variation metrics per Rule 14 and Table 1.

    The returned index combines:
    - HSV color dispersion
    - Hue/saturation histogram diversity
    - Significant dominant-color regions
    """

    default_res = {
        "h": 0.0,
        "s": 0.0,
        "v": 0.0,
        "h_raw": 0.0,
        "s_raw": 0.0,
        "v_raw": 0.0,
        "index": 0.0,
        "hue_mean": 0.0,
        "hue_variation": 0.0,
        "saturation_mean": 0.0,
        "saturation_variation": 0.0,
        "brightness_mean": 0.0,
        "brightness_variation": 0.0,
        "color_diversity": 0.0,
        "dominant_colors": [],
    }

    # Validate inputs.
    if (
        image is None
        or mask is None
        or np.count_nonzero(mask) == 0
    ):
        return default_res

    # Convert RGB image to HSV.
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV,
    ).astype(np.float32)

    # Keep only lesion pixels.
    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        return default_res

    # ---------------------------------------------------------
    # HSV means and standard deviations
    # ---------------------------------------------------------

    hue_mean = float(np.mean(pixels[:, 0]))
    hue_std = float(np.std(pixels[:, 0]))

    sat_mean = float(np.mean(pixels[:, 1]))
    sat_std = float(np.std(pixels[:, 1]))

    val_mean = float(np.mean(pixels[:, 2]))
    val_std = float(np.std(pixels[:, 2]))

    # ---------------------------------------------------------
    # Normalized HSV variations
    # ---------------------------------------------------------

    h_norm = float(
        np.clip(
            hue_std / _H_STD_MAX,
            0.0,
            1.0,
        )
    )

    s_norm = float(
        np.clip(
            sat_std / _SV_STD_MAX,
            0.0,
            1.0,
        )
    )

    v_norm = float(
        np.clip(
            val_std / _SV_STD_MAX,
            0.0,
            1.0,
        )
    )

    # ---------------------------------------------------------
    # Color diversity using hue-saturation histogram entropy
    # ---------------------------------------------------------

    h_bins = 18
    s_bins = 16

    hist, _, _ = np.histogram2d(
        pixels[:, 0],
        pixels[:, 1],
        bins=[h_bins, s_bins],
        range=[
            [0, 180],
            [0, 256],
        ],
    )

    hist_prob = hist / (hist.sum() + 1e-7)

    non_zero = hist_prob[hist_prob > 0]

    entropy = -float(
        np.sum(
            non_zero * np.log2(non_zero)
        )
    )

    max_entropy = np.log2(h_bins * s_bins)

    color_diversity = float(
        np.clip(
            entropy / max_entropy,
            0.0,
            1.0,
        )
    )

    # ---------------------------------------------------------
    # Dominant colors using k-means clustering in RGB space
    # ---------------------------------------------------------

    rgb_pixels = image[mask > 0].astype(np.float32)

    dominant_colors = []

    if len(rgb_pixels) >= 3:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0,
        )

        k = min(3, len(rgb_pixels))

        _, labels, centers = cv2.kmeans(
            rgb_pixels,
            k,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )

        counts = np.bincount(
            labels.flatten()
        )

        total = float(len(labels))

        for center, count in sorted(
            zip(centers, counts),
            key=lambda x: x[1],
            reverse=True,
        ):

            r, g, b = (
                int(center[0]),
                int(center[1]),
                int(center[2]),
            )

            pct = round(
                float(count) / total,
                3,
            )

            hex_code = (
                f"#{r:02x}{g:02x}{b:02x}"
            )

            dominant_colors.append(
                {
                    "hex": hex_code,
                    "rgb": [r, g, b],
                    "percentage": pct,
                }
            )

    # ---------------------------------------------------------
    # Dominant-color diversity
    # ---------------------------------------------------------

    dominant_color_diversity = 0.0

    if len(dominant_colors) >= 2:

        significant_colors = [
            item
            for item in dominant_colors
            if float(item["percentage"]) >= 0.05
        ]

        # Multiple substantial color regions indicate
        # meaningful within-lesion color diversity.
        if len(significant_colors) >= 2:

            dominant_color_diversity = min(
                1.0,
                (len(significant_colors) - 1) / 2.0,
            )

    # Combine histogram-based diversity with
    # dominant-color diversity.
    combined_diversity = max(
        color_diversity,
        dominant_color_diversity,
    )

    # ---------------------------------------------------------
    # Composite color variation index
    # ---------------------------------------------------------

    hsv_variation = (
        h_norm
        + s_norm
        + v_norm
    ) / 3.0

    color_index = float(
        np.clip(
            0.55 * hsv_variation
            + 0.45 * combined_diversity,
            0.0,
            1.0,
        )
    )

    # Preserve strong-variation safeguard.
    if (
        hue_std > 15.0
        or sat_std > 35.0
        or val_std > 40.0
    ):
        color_index = max(
            color_index,
            0.35,
        )

    # ---------------------------------------------------------
    # Return complete feature set
    # ---------------------------------------------------------

    return {
        # Legacy keys for backward compatibility
        "h": round(h_norm, 4),
        "s": round(s_norm, 4),
        "v": round(v_norm, 4),

        "h_raw": round(hue_std, 4),
        "s_raw": round(sat_std, 4),
        "v_raw": round(val_std, 4),

        "index": round(
            color_index,
            4,
        ),

        # Rule 14 keys
        "hue_mean": round(
            hue_mean,
            2,
        ),

        "hue_variation": round(
            hue_std,
            4,
        ),

        "saturation_mean": round(
            sat_mean,
            2,
        ),

        "saturation_variation": round(
            sat_std,
            4,
        ),

        "brightness_mean": round(
            val_mean,
            2,
        ),

        "brightness_variation": round(
            val_std,
            4,
        ),

        "color_diversity": round(
            combined_diversity,
            4,
        ),

        "dominant_colors": dominant_colors,
    }