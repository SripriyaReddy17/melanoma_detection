import json
from pathlib import Path

import cv2
import numpy as np

from app.services.segmentation import segment_lesion


BASE_DIR = Path("/mnt/d/pytest_cache")

IMAGE_DIR = (
    BASE_DIR
    / "datasets"
    / "ISIC2018"
    / "ISIC2018_Task1-2_Training_Input"
)

MASK_DIR = (
    BASE_DIR
    / "datasets"
    / "ISIC2018"
    / "ISIC2018_Task1_Training_GroundTruth"
)

SPLIT_FILE = (
    BASE_DIR
    / "outputs"
    / "segmentation"
    / "segmentation_split.json"
)


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()

    return (2.0 * intersection) / (
        pred.sum() + gt.sum() + 1e-8
    )


def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return intersection / (union + 1e-8)


def calculate_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return dice, iou, precision, recall, specificity, accuracy


# --------------------------------------------------
# Load validation split
# --------------------------------------------------

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_images = split["validation_images"]

print("=" * 60)
print("APPLICATION VM-UNET VALIDATION")
print("=" * 60)

print(f"Validation images: {len(validation_images)}")
print("Using actual app/services/segmentation.py")
print("Using segment_lesion()")
print("=" * 60)


all_metrics = []

for index, filename in enumerate(validation_images, start=1):

    image_path = IMAGE_DIR / filename

    # Ground-truth filenames are usually .png
    stem = Path(filename).stem
    mask_path = MASK_DIR / f"{stem}_segmentation.png"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if gt is None:
        print(f"WARNING: Could not read {mask_path}")
        continue

    # ---------------------------------------------
    # ACTUAL APPLICATION SEGMENTATION
    # ---------------------------------------------

    pred = segment_lesion(str(image_path))

    # Ensure same dimensions
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    pred = pred > 0
    gt = gt > 127

    metrics = calculate_metrics(pred, gt)

    all_metrics.append(metrics)

    if index % 50 == 0 or index == 1:
        print(
            f"[{index}/{len(validation_images)}] "
            f"Dice: {metrics[0]:.4f} | "
            f"IoU: {metrics[1]:.4f}"
        )


# --------------------------------------------------
# Average metrics
# --------------------------------------------------

all_metrics = np.array(all_metrics)

mean_metrics = all_metrics.mean(axis=0)

dice, iou, precision, recall, specificity, accuracy = mean_metrics


print()
print("=" * 60)
print("FINAL APPLICATION VALIDATION RESULTS")
print("=" * 60)

print(f"Images evaluated : {len(all_metrics)}")
print(f"Dice             : {dice:.4f} ({dice * 100:.2f}%)")
print(f"IoU              : {iou:.4f} ({iou * 100:.2f}%)")
print(f"Precision        : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall           : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Specificity      : {specificity:.4f} ({specificity * 100:.2f}%)")
print(f"Pixel Accuracy   : {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("=" * 60)
