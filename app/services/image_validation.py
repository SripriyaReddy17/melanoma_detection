from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def validate_file_extension(filename: str) -> tuple[bool, str]:
    """
    Check whether the uploaded file has a supported extension.
    """

    if not filename:
        return False, "Filename is missing."

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return False, (
            "Unsupported image format. "
            "Please upload JPG, JPEG, or PNG."
        )

    return True, "File format is supported."


def load_image(image_path: str):
    """
    Load image using OpenCV.
    Returns RGB image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read the uploaded image.")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def check_resolution(image: np.ndarray) -> tuple[bool, str]:
    """
    Check minimum image resolution.
    """

    height, width = image.shape[:2]

    if width < 224 or height < 224:
        return False, (
            f"Image resolution is too low "
            f"({width}x{height}). "
            f"Minimum required resolution is 224x224."
        )

    return True, f"Resolution acceptable ({width}x{height})."


def check_brightness(image: np.ndarray) -> tuple[bool, str]:
    """
    Detect extremely dark or overexposed images.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    brightness = float(np.mean(gray))

    if brightness < 20:
        return False, "Image is too dark."

    if brightness > 245:
        return False, "Image is overexposed."

    return True, f"Brightness acceptable ({brightness:.2f})."


def check_blur(image: np.ndarray) -> tuple[bool, str]:
    """
    Estimate image sharpness using Laplacian variance.

    NOTE:
        The threshold is only a preliminary quality-control
        threshold and must be calibrated using the actual
        project datasets.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # Preliminary threshold.
    # This will be calibrated later using ISIC/HAM10000
    # and clinical-image samples.
    BLUR_THRESHOLD = 10.0

    if variance < BLUR_THRESHOLD:
        return False, (
            f"Image appears very blurry "
            f"(sharpness={variance:.2f})."
        )

    return True, (
        f"Image sharpness acceptable "
        f"({variance:.2f})."
    )


def check_image_channels(image: np.ndarray) -> tuple[bool, str]:
    """
    Make sure the image has three color channels.
    """

    if len(image.shape) != 3:
        return False, "Image does not contain RGB color channels."

    if image.shape[2] != 3:
        return False, "Image must contain three color channels."

    return True, "RGB image confirmed."


def check_reference_object(image: np.ndarray) -> tuple[bool, str, dict]:
    """
    Legacy hook.

    Strict marker validation is performed after segmentation
    by the calibration/post-segmentation quality pipeline.
    """

    return (
        True,
        "A 20 mm marker or 1 mm graduated ruler will be "
        "validated during calibration.",
        {}
    )


def check_post_segmentation_quality(
    image: np.ndarray,
    mask: np.ndarray,
    calibration: dict
) -> dict:
    """
    Strict post-segmentation quality checks.

    Checks:
        1. Valid lesion segmentation
        2. Lesion does not reach image boundary
        3. Plausible lesion area
        4. Lesion sharpness
        5. Lesion glare/overexposure
        6. Physical calibration marker/ruler

    A valid calibration marker/ruler is required for this
    post-segmentation quality gate.
    """

    checks = {}
    valid = True
    reasons = []

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # 1. Segmentation
    # --------------------------------------------------------

    if mask is None or np.count_nonzero(mask) == 0:
        checks["segmentation"] = {
            "valid": False,
            "message": "Lesion could not be reliably segmented."
        }

        return {
            "valid": False,
            "checks": checks,
            "reason": checks["segmentation"]["message"]
        }

    # --------------------------------------------------------
    # 2. Lesion boundary
    # --------------------------------------------------------

    ys, xs = np.where(mask > 0)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # A segmentation mask reaching the image edge is treated
    # as an unreliable lesion isolation result.
    #
    # We deliberately do NOT claim that the patient's lesion
    # itself is cropped, because a segmentation error can also
    # cause the mask to reach the boundary.
    if x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2:
        checks["lesion_boundary"] = {
            "valid": False,
            "message": (
                "Lesion could not be isolated reliably because "
                "the detected lesion region reaches the image "
                "boundary. Please retake the image with the "
                "complete lesion clearly visible."
            )
        }

        valid = False
        reasons.append(
            checks["lesion_boundary"]["message"]
        )

    else:
        checks["lesion_boundary"] = {
            "valid": True,
            "message": (
                "Detected lesion region is fully inside the image."
            )
        }

    # --------------------------------------------------------
    # 3. Lesion area
    # --------------------------------------------------------

    area = float(np.count_nonzero(mask))
    fraction = area / (w * h)

    if fraction < 0.0002 or fraction > 0.60:
        checks["lesion_area"] = {
            "valid": False,
            "message": (
                "Lesion segmentation is not reliable "
                f"(area fraction={fraction:.4f})."
            )
        }

        valid = False
        reasons.append(
            checks["lesion_area"]["message"]
        )

    else:
        checks["lesion_area"] = {
            "valid": True,
            "message": (
                f"Lesion area is plausible "
                f"({fraction:.4f} of image)."
            )
        }

    # --------------------------------------------------------
    # 4. Lesion sharpness and glare
    # --------------------------------------------------------

    roi = image[
        max(0, y0):min(h, y1 + 1),
        max(0, x0):min(w, x1 + 1)
    ]

    if roi.size:

        # ----------------------------------------------------
        # Lesion sharpness
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2GRAY
        )

        sharp = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        if sharp < 12.0:
            checks["lesion_sharpness"] = {
                "valid": False,
                "message": (
                    f"Lesion region is too blurry "
                    f"(sharpness={sharp:.2f})."
                )
            }

            valid = False
            reasons.append(
                checks["lesion_sharpness"]["message"]
            )

        else:
            checks["lesion_sharpness"] = {
                "valid": True,
                "message": (
                    f"Lesion sharpness acceptable "
                    f"({sharp:.2f})."
                )
            }

        # ----------------------------------------------------
        # Glare / overexposure
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2HSV
        )

        glare = float(
            np.mean(
                (hsv[:, :, 1] < 25)
                & (hsv[:, :, 2] > 245)
            )
        )

        if glare > 0.08:
            checks["glare"] = {
                "valid": False,
                "message": (
                    "Strong glare/overexposure detected "
                    f"in lesion region ({glare:.1%})."
                )
            }

            valid = False
            reasons.append(
                checks["glare"]["message"]
            )

        else:
            checks["glare"] = {
                "valid": True,
                "message": (
                    "No excessive glare detected "
                    "in lesion region."
                )
            }

    # --------------------------------------------------------
    # 5. Physical calibration
    # --------------------------------------------------------

    marker_valid = bool(
        calibration
        and calibration.get("valid")
        and calibration.get("pixels_per_mm")
    )

    checks["calibration"] = {
        "valid": marker_valid,
        "message": (
            calibration.get(
                "message",
                "Calibration marker detected."
            )
            if marker_valid
            else (
                "Physical calibration marker/ruler was not "
                "detected. A calibrated diameter measurement "
                "cannot be performed."
            )
        )
    }

    # Missing calibration fails the post-segmentation
    # quality gate.
    if not marker_valid:
        valid = False
        reasons.append(
            "Physical calibration marker/ruler was not detected."
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "valid": valid,
        "checks": checks,
        "calibration_used": marker_valid,
        "reason": (
            " ".join(reasons)
            if reasons
            else "All post-segmentation quality checks passed."
        )
    }


def validate_image(
    image_path: str,
    filename: str = ""
) -> dict:
    """
    Complete basic image validation.

    This function checks:
        1. File extension
        2. Image readability
        3. Resolution
        4. Brightness
        5. Blur
        6. RGB channels

    It does NOT determine whether the image is dermoscopic.
    """

    results = {}

    # --------------------------------------------------------
    # 1. File extension
    # --------------------------------------------------------

    extension_valid, extension_message = validate_file_extension(
        filename or image_path
    )

    results["file_format"] = {
        "valid": extension_valid,
        "message": extension_message
    }

    if not extension_valid:
        return {
            "valid": False,
            "checks": results,
            "reason": extension_message
        }

    # --------------------------------------------------------
    # 2. Load image
    # --------------------------------------------------------

    try:
        image = load_image(image_path)

    except Exception as e:
        return {
            "valid": False,
            "checks": results,
            "reason": str(e)
        }

    # --------------------------------------------------------
    # 3. Resolution
    # --------------------------------------------------------

    resolution_valid, resolution_message = check_resolution(
        image
    )

    results["resolution"] = {
        "valid": resolution_valid,
        "message": resolution_message
    }

    # --------------------------------------------------------
    # 4. Brightness
    # --------------------------------------------------------

    brightness_valid, brightness_message = check_brightness(
        image
    )

    results["brightness"] = {
        "valid": brightness_valid,
        "message": brightness_message
    }

    # --------------------------------------------------------
    # 5. Blur
    # --------------------------------------------------------

    blur_valid, blur_message = check_blur(
        image
    )

    results["sharpness"] = {
        "valid": blur_valid,
        "message": blur_message
    }

    # --------------------------------------------------------
    # 6. Channels
    # --------------------------------------------------------

    channels_valid, channels_message = check_image_channels(
        image
    )

    results["channels"] = {
        "valid": channels_valid,
        "message": channels_message
    }

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    checks = [
        resolution_valid,
        brightness_valid,
        blur_valid,
        channels_valid
    ]

    all_valid = all(checks)

    if all_valid:
        reason = "All basic image quality checks passed."

    else:
        failed_checks = [
            value["message"]
            for value in results.values()
            if not value["valid"]
        ]

        reason = " ".join(failed_checks)

    height, width = image.shape[:2]

    format_name = (
        Path(filename or image_path)
        .suffix
        .lstrip(".")
        .lower()
        or "jpg"
    )

    aspect_ratio = round(
        width / max(1, height),
        3
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    return {
        "valid": all_valid,
        "format": format_name,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "quality": {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "is_blurry": not blur_valid,
        },
        "checks": results,
        "calibration": None,
        "reason": reason,
    }