import os
import json
import cv2
import torch
import numpy as np

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet


# ============================================================
# PATHS
# ============================================================

ROOT = "/mnt/d/pytest_cache"

IMAGE_DIR = os.path.join(
    ROOT,
    "datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"
)

SPLIT_FILE = os.path.join(
    ROOT,
    "outputs/segmentation/segmentation_split.json"
)

MODEL_PATH = os.path.join(
    ROOT,
    "models/segmentation/UltraLight_VM_UNet.pth"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs/sam2_batch/uncertainty_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.1


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("VM-UNet UNCERTAINTY ANALYSIS")
print("=" * 70)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_images = split[
    "validation_images"
][:20]

print(
    "Validation images:",
    len(validation_images)
)


# ============================================================
# LOAD MODEL
# ============================================================

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8, 16, 24, 32, 48, 64]
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(
    checkpoint,
    strict=True
)

model.to(device)
model.eval()

print("VM-UNet loaded successfully.")


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    h, w = image.shape[:2]

    resized = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(
        2, 0, 1
    ).unsqueeze(0)

    return image, tensor, h, w


# ============================================================
# ENTROPY
# ============================================================

def binary_entropy(probability):

    p = np.clip(
        probability,
        1e-7,
        1.0 - 1e-7
    )

    entropy = (
        -p * np.log2(p)
        -(1.0 - p) *
        np.log2(1.0 - p)
    )

    return entropy


# ============================================================
# PROCESS
# ============================================================

records = []

with torch.no_grad():

    for index, filename in enumerate(
        validation_images,
        start=1
    ):

        print()
        print(
            f"[{index}/20] {filename}"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        image, tensor, h, w = load_image(
            image_path
        )

        tensor = tensor.to(device)

        # VM-UNet already applies sigmoid.
        prediction = model(
            tensor
        )

        prediction = (
            prediction
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # Resize probability map to original image.
        probability = cv2.resize(
            prediction,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Binary mask using the EXISTING threshold.
        # ----------------------------------------------------

        binary = (
            probability >= THRESHOLD
        )

        # ----------------------------------------------------
        # Basic statistics
        # ----------------------------------------------------

        total_pixels = binary.size

        foreground_ratio = (
            binary.sum() /
            total_pixels
        )

        # ----------------------------------------------------
        # Mean probability
        # ----------------------------------------------------

        mean_probability = (
            probability.mean()
        )

        # ----------------------------------------------------
        # Confidence
        #
        # Distance from 0.5.
        #
        # 0.5 = uncertain
        # 0 or 1 = confident
        #
        # Range: 0 to 1.
        # ----------------------------------------------------

        confidence = (
            np.abs(
                probability - 0.5
            ) * 2.0
        )

        mean_confidence = (
            confidence.mean()
        )

        # ----------------------------------------------------
        # Uncertainty
        #
        # Pixels close to 0.5.
        # ----------------------------------------------------

        uncertain_05 = (
            np.abs(
                probability - 0.5
            ) < 0.05
        )

        uncertain_10 = (
            np.abs(
                probability - 0.5
            ) < 0.10
        )

        uncertain_ratio_05 = (
            uncertain_05.sum() /
            total_pixels
        )

        uncertain_ratio_10 = (
            uncertain_10.sum() /
            total_pixels
        )

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        entropy = binary_entropy(
            probability
        )

        mean_entropy = (
            entropy.mean()
        )

        # ----------------------------------------------------
        # Foreground confidence
        #
        # Only pixels classified as foreground.
        # ----------------------------------------------------

        if binary.any():

            foreground_probability = (
                probability[binary]
            )

            mean_foreground_probability = (
                foreground_probability.mean()
            )

            foreground_confidence = (
                confidence[binary].mean()
            )

        else:

            mean_foreground_probability = 0.0
            foreground_confidence = 0.0

        # ----------------------------------------------------
        # Background confidence
        # ----------------------------------------------------

        background = ~binary

        if background.any():

            mean_background_probability = (
                probability[background].mean()
            )

            background_confidence = (
                confidence[background].mean()
            )

        else:

            mean_background_probability = 1.0
            background_confidence = 0.0

        # ----------------------------------------------------
        # Save probability map
        # ----------------------------------------------------

        probability_path = os.path.join(
            OUTPUT_DIR,
            filename.replace(
                ".jpg",
                ".npy"
            )
        )

        np.save(
            probability_path,
            probability.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {

            "image":
                filename,

            "height":
                int(h),

            "width":
                int(w),

            "foreground_ratio":
                float(foreground_ratio),

            "mean_probability":
                float(mean_probability),

            "mean_confidence":
                float(mean_confidence),

            "uncertain_ratio_05":
                float(uncertain_ratio_05),

            "uncertain_ratio_10":
                float(uncertain_ratio_10),

            "mean_entropy":
                float(mean_entropy),

            "mean_foreground_probability":
                float(
                    mean_foreground_probability
                ),

            "foreground_confidence":
                float(
                    foreground_confidence
                ),

            "mean_background_probability":
                float(
                    mean_background_probability
                ),

            "background_confidence":
                float(
                    background_confidence
                ),

            "probability_map":
                probability_path
        }

        records.append(record)

        print(
            f"  Foreground ratio : "
            f"{foreground_ratio:.4f}"
        )

        print(
            f"  Mean probability : "
            f"{mean_probability:.4f}"
        )

        print(
            f"  Mean confidence  : "
            f"{mean_confidence:.4f}"
        )

        print(
            f"  Uncertain <0.05  : "
            f"{uncertain_ratio_05:.4f}"
        )

        print(
            f"  Uncertain <0.10  : "
            f"{uncertain_ratio_10:.4f}"
        )

        print(
            f"  Mean entropy     : "
            f"{mean_entropy:.4f}"
        )


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "vmunet_uncertainty_20.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        records,
        f,
        indent=2
    )


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "vmunet_uncertainty_20.csv"
)

import csv

fields = [
    "image",
    "height",
    "width",
    "foreground_ratio",
    "mean_probability",
    "mean_confidence",
    "uncertain_ratio_05",
    "uncertain_ratio_10",
    "mean_entropy",
    "mean_foreground_probability",
    "foreground_confidence",
    "mean_background_probability",
    "background_confidence"
]

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for record in records:

        writer.writerow({
            field: record[field]
            for field in fields
        })


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "JSON:",
    json_path
)

print(
    "CSV:",
    csv_path
)

print(
    "Probability maps:",
    OUTPUT_DIR
)
