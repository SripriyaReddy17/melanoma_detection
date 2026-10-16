import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = str(
    BASE_DIR / "datasets" / "ISIC2018" /
    "ISIC2018_Task1-2_Training_Input"
)
MASK_DIR = str(
    BASE_DIR / "datasets" / "ISIC2018" /
    "ISIC2018_Task1_Training_GroundTruth"
)
MODEL_PATH = str(
    BASE_DIR / "models" / "segmentation" /
    "UltraLight_VM_UNet.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL
# ============================================================

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8, 16, 24, 32, 48, 64]
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
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
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}


missing, unexpected = model.load_state_dict(
    state_dict,
    strict=True
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


model = model.to(DEVICE)
model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + 1e-7
    )

    iou = (
        tp
    ) / (
        tp + fp + fn + 1e-7
    )

    precision = (
        tp
    ) / (
        tp + fp + 1e-7
    )

    recall = (
        tp
    ) / (
        tp + fn + 1e-7
    )

    specificity = (
        tn
    ) / (
        tn + fp + 1e-7
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + 1e-7
    )

    return (
        dice,
        iou,
        precision,
        recall,
        specificity,
        accuracy
    )


# ============================================================
# FIND IMAGES
# ============================================================

image_files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])

print("Images found:", len(image_files))


# ============================================================
# METRIC STORAGE
# ============================================================

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
specificity_scores = []
accuracy_scores = []


# ============================================================
# THRESHOLD ANALYSIS
#
# This is for understanding the model output.
#
# IMPORTANT:
# The final evaluation below still uses a fixed threshold.
# ============================================================

thresholds = np.arange(
    0.1,
    1.0,
    0.1
)

threshold_dice = {
    float(t): []
    for t in thresholds
}


# ============================================================
# PROCESS DATASET
# ============================================================

with torch.no_grad():

    for idx, filename in enumerate(image_files):

        image_id = os.path.splitext(filename)[0]

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        mask_path = os.path.join(
            MASK_DIR,
            image_id + "_segmentation.png"
        )

        if not os.path.exists(mask_path):

            print(
                "Missing mask:",
                filename
            )

            continue


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")


        # ----------------------------------------------------
        # IMAGE TENSOR
        # ----------------------------------------------------

        image_tensor = image_transform(
            image
        ).unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # GROUND TRUTH MASK
        # ----------------------------------------------------

        mask = mask.resize(
            (224, 224),
            Image.NEAREST
        )

        mask = np.array(mask)

        target = mask > 0


        # ----------------------------------------------------
        # MODEL FORWARD PASS
        # ----------------------------------------------------

        output = model(image_tensor)


        # ----------------------------------------------------
        # REMOVE EXTRA DIMENSIONS
        # ----------------------------------------------------

        raw_output = output.squeeze().cpu().numpy()


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Check whether model output already looks like
        # probabilities.
        #
        # If values are between 0 and 1, do NOT blindly
        # apply sigmoid.
        # ----------------------------------------------------

        raw_min = float(raw_output.min())
        raw_max = float(raw_output.max())


        if raw_min >= 0.0 and raw_max <= 1.0:

            probability = raw_output

            output_type = "probability"

        else:

            probability = 1.0 / (
                1.0 + np.exp(-raw_output)
            )

            output_type = "logit + sigmoid"


        # ----------------------------------------------------
        # THRESHOLD ANALYSIS
        # ----------------------------------------------------

        for threshold in thresholds:

            test_pred = probability >= threshold

            (
                dice,
                iou,
                precision,
                recall,
                specificity,
                accuracy
            ) = calculate_metrics(
                test_pred,
                target
            )

            threshold_dice[
                float(threshold)
            ].append(dice)


        # ----------------------------------------------------
        # FINAL EVALUATION THRESHOLD
        #
        # Keep this fixed for fair evaluation.
        # ----------------------------------------------------

        FINAL_THRESHOLD = 0.5

        pred = probability >= FINAL_THRESHOLD


        # ----------------------------------------------------
        # FINAL METRICS
        # ----------------------------------------------------

        (
            dice,
            iou,
            precision,
            recall,
            specificity,
            accuracy
        ) = calculate_metrics(
            pred,
            target
        )


        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        specificity_scores.append(specificity)
        accuracy_scores.append(accuracy)


        # ----------------------------------------------------
        # PRINT EVERY 100 IMAGES
        # ----------------------------------------------------

        if (idx + 1) % 100 == 0:

            foreground_pixels = pred.sum()

            total_pixels = pred.size

            print(
                f"[{idx + 1}/{len(image_files)}] "
                f"Dice={dice:.4f} "
                f"IoU={iou:.4f} "
                f"Foreground="
                f"{foreground_pixels / total_pixels * 100:.2f}% "
                f"Output={output_type}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 65)
print("ULTRALIGHT VM-UNET SEGMENTATION RESULTS")
print("=" * 65)

print(
    f"Images evaluated : {len(dice_scores)}"
)

print(
    f"Dice             : "
    f"{np.mean(dice_scores):.4f} "
    f"({np.mean(dice_scores) * 100:.2f}%)"
)

print(
    f"IoU              : "
    f"{np.mean(iou_scores):.4f} "
    f"({np.mean(iou_scores) * 100:.2f}%)"
)

print(
    f"Precision        : "
    f"{np.mean(precision_scores):.4f} "
    f"({np.mean(precision_scores) * 100:.2f}%)"
)

print(
    f"Recall           : "
    f"{np.mean(recall_scores):.4f} "
    f"({np.mean(recall_scores) * 100:.2f}%)"
)

print(
    f"Specificity      : "
    f"{np.mean(specificity_scores):.4f} "
    f"({np.mean(specificity_scores) * 100:.2f}%)"
)

print(
    f"Pixel Accuracy   : "
    f"{np.mean(accuracy_scores):.4f} "
    f"({np.mean(accuracy_scores) * 100:.2f}%)"
)

print("=" * 65)


# ============================================================
# THRESHOLD ANALYSIS RESULTS
# ============================================================

print()
print("=" * 65)
print("THRESHOLD ANALYSIS")
print("=" * 65)

best_threshold = None
best_threshold_dice = -1

for threshold in thresholds:

    mean_dice = np.mean(
        threshold_dice[
            float(threshold)
        ]
    )

    print(
        f"Threshold {threshold:.1f} "
        f"-> Mean Dice = {mean_dice:.4f}"
    )

    if mean_dice > best_threshold_dice:

        best_threshold_dice = mean_dice
        best_threshold = threshold


print()
print(
    f"Best threshold in analysis : "
    f"{best_threshold:.1f}"
)

print(
    f"Best threshold Dice        : "
    f"{best_threshold_dice:.4f}"
)

print()
print(
    "NOTE: The best threshold above is for diagnosis/"
    "threshold analysis."
)

print(
    "For a fair final test-set evaluation, choose the "
    "threshold using a validation set and then keep it "
    "fixed on the test set."
)

print("=" * 65)