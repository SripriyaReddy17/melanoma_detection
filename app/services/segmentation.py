"""
Lesion segmentation service for CEFM.

UltraLight VM-UNet trained on ISIC2018 is the only lesion
segmentation source. Segmentation failures are reported explicitly.
"""

from pathlib import Path
import cv2
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SEGMENTATION_WEIGHTS = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "UltraLight_VM_UNet.pth"
)


# ============================================================
# MODEL CACHE
# ============================================================

_vmunet_model = None


class SegmentationError(RuntimeError):
    """Raised when VM-UNet cannot produce a lesion segmentation mask."""


# ============================================================
# LOAD ULTRALIGHT VM-UNET
# ============================================================

def _get_vmunet_model():

    global _vmunet_model

    if _vmunet_model is not None:
        return _vmunet_model

    if not SEGMENTATION_WEIGHTS.exists():
        raise SegmentationError(
            f"UltraLight VM-UNet checkpoint not found: "
            f"{SEGMENTATION_WEIGHTS}"
        )

    try:

        from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import (
            UltraLight_VM_UNet
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = UltraLight_VM_UNet(
            num_classes=1,
            input_channels=3,
            c_list=[8, 16, 24, 32, 48, 64]
        )

        checkpoint = torch.load(
            SEGMENTATION_WEIGHTS,
            map_location=device
        )

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint


        # Remove DataParallel prefix if present
        state_dict = {
            key.replace("module.", "", 1)
            if key.startswith("module.")
            else key: value

            for key, value in state_dict.items()
        }


        model.load_state_dict(
            state_dict,
            strict=True
        )

        model = model.to(device)
        model.eval()

        _vmunet_model = model

        print(
            f"[segmentation] Loaded UltraLight VM-UNet "
            f"from {SEGMENTATION_WEIGHTS}"
        )

        print(
            f"[segmentation] Device: {device}"
        )

        return _vmunet_model

    except Exception as exc:
        raise SegmentationError(
            f"Could not load UltraLight VM-UNet checkpoint "
            f"{SEGMENTATION_WEIGHTS}: {exc}"
        ) from exc


# ============================================================
# HAIR REMOVAL
# ============================================================

def remove_hair(
    image_bgr: np.ndarray
) -> np.ndarray:

    """
    DullRazor-style hair removal:
    blackhat morphology -> hair mask -> inpaint.
    """

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, hair_mask = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    hair_mask = cv2.dilate(
        hair_mask,
        np.ones((3, 3), np.uint8),
        iterations=2
    )

    if cv2.countNonZero(hair_mask) == 0:
        return image_bgr

    return cv2.inpaint(
        image_bgr,
        hair_mask,
        5,
        cv2.INPAINT_TELEA
    )


# ============================================================
# MAIN SEGMENTATION
# ============================================================

def segment_lesion(
    image_path: str | Path,
    remove_hair_flag: bool = False,
    exclude_polygon=None,
) -> np.ndarray:

    """
    Main lesion segmentation function.

    Primary:
        UltraLight VM-UNet

    Input:
        Original image

    Processing:
        Resize to 224x224
        UltraLight VM-UNet inference
        Model already applies sigmoid internally
        Threshold = 0.1

    Output:
        Binary mask with values 0 or 1.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise SegmentationError(
            f"Could not load image for VM-UNet segmentation: "
            f"{image_path}"
        )


    if remove_hair_flag:

        image = remove_hair(
            image
        )


    h, w = image.shape[:2]


    # ========================================================
    # ULTRALIGHT VM-UNET
    # ========================================================

    vmunet = _get_vmunet_model()

    try:
        # BGR → RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Same 224x224 preprocessing used in evaluation
        resized = cv2.resize(
            rgb,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )

        # HWC → CHW
        tensor_in = torch.from_numpy(
            resized.transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0

        device = next(
            vmunet.parameters()
        ).device

        tensor_in = tensor_in.to(
            device
        )

        with torch.no_grad():
            output = vmunet(tensor_in)

        # VM-UNet applies sigmoid internally.
        prediction = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )

        mask_224 = (
            prediction >= 0.1
        ).astype(np.uint8)

        # Restore original image size
        mask = cv2.resize(
            mask_224,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        if exclude_polygon is not None:
            poly = np.asarray(
                exclude_polygon,
                dtype=np.int32
            ).reshape(-1, 2)

            if len(poly) >= 3:
                cv2.fillPoly(
                    mask,
                    [poly],
                    0
                )

        # Keep the largest connected VM-UNet component.
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(
                contours,
                key=cv2.contourArea
            )
            clean = np.zeros_like(mask)
            cv2.drawContours(
                clean,
                [largest],
                -1,
                1,
                cv2.FILLED
            )
            mask = clean

        return mask.astype(np.uint8)

    except Exception as exc:
        raise SegmentationError(
            f"UltraLight VM-UNet inference failed: {exc}"
        ) from exc
