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

GT_DIR = os.path.join(
    ROOT,
    "datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
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
    "outputs/sam2_batch/postprocessing_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.1

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("VM-UNET POST-PROCESSING EVALUATION")
print("=" * 75)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD VALIDATION SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_images = split["validation_images"]

print(f"Validation images: {len(validation_images)}")


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

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint, strict=True)

model.to(device)
model.eval()

print("VM-UNet loaded successfully.")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return image, tensor, original_h, original_w


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    union = np.logical_or(pred, gt).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    denominator_dice = 2 * tp + fp + fn

    if denominator_dice == 0:
        dice = 1.0
    else:
        dice = 2 * tp / denominator_dice

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy)
    }


# ============================================================
# POST-PROCESSING METHODS
# ============================================================

def baseline(mask):
    return mask.copy()


def opening(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


def closing(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


def opening_closing(mask):

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    result = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    return result


def fill_holes(mask):

    binary = (mask > 0).astype(np.uint8)

    flood = binary.copy()

    h, w = binary.shape

    flood_mask = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        2
    )

    outside = (flood == 2).astype(np.uint8)

    holes = (
        (binary == 0) &
        (outside == 0)
    ).astype(np.uint8)

    result = binary | holes

    return result


def largest_component(mask):

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = 1 + np.argmax(areas)

    result = (
        labels == largest_label
    ).astype(np.uint8)

    return result


def closing_holes(mask):

    result = closing(mask)
    result = fill_holes(result)

    return result


def largest_holes(mask):

    result = largest_component(mask)
    result = fill_holes(result)

    return result


def opening_closing_holes(mask):

    result = opening_closing(mask)
    result = fill_holes(result)

    return result


METHODS = {
    "baseline": baseline,
    "opening": opening,
    "closing": closing,
    "opening_closing": opening_closing,
    "fill_holes": fill_holes,
    "largest_component": largest_component,
    "closing_holes": closing_holes,
    "largest_holes": largest_holes,
    "opening_closing_holes": opening_closing_holes,
}


# ============================================================
# STORAGE
# ============================================================

all_results = {
    method: []
    for method in METHODS
}


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for index, filename in enumerate(
        validation_images,
        start=1
    ):

        print(
            f"\n[{index}/{len(validation_images)}] {filename}"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        gt_name = filename.replace(
            ".jpg",
            "_segmentation.png"
        )

        gt_path = os.path.join(
            GT_DIR,
            gt_name
        )

        image, tensor, h, w = load_image(
            image_path
        )

        tensor = tensor.to(device)

        prediction = model(tensor)

        prediction = (
            prediction
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        probability = cv2.resize(
            prediction,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        base_mask = (
            probability >= THRESHOLD
        ).astype(np.uint8)

        gt = cv2.imread(
            gt_path,
            cv2.IMREAD_GRAYSCALE
        )

        if gt is None:
            print(
                f"WARNING: GT missing: {gt_path}"
            )
            continue

        gt = (
            gt > 127
        ).astype(np.uint8)

        for method_name, method_function in METHODS.items():

            processed = method_function(
                base_mask
            )

            metrics = calculate_metrics(
                processed,
                gt
            )

            all_results[method_name].append(
                metrics
            )


# ============================================================
# AVERAGE RESULTS
# ============================================================

summary = []

for method_name, results in all_results.items():

    if len(results) == 0:
        continue

    summary.append({
        "method": method_name,
        "images": len(results),

        "iou": np.mean([
            r["iou"]
            for r in results
        ]),

        "dice": np.mean([
            r["dice"]
            for r in results
        ]),

        "precision": np.mean([
            r["precision"]
            for r in results
        ]),

        "recall": np.mean([
            r["recall"]
            for r in results
        ]),

        "specificity": np.mean([
            r["specificity"]
            for r in results
        ]),

        "accuracy": np.mean([
            r["accuracy"]
            for r in results
        ])
    })


# ============================================================
# SORT BY IOU
# ============================================================

summary.sort(
    key=lambda x: x["iou"],
    reverse=True
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("POST-PROCESSING RESULTS")
print("=" * 75)

print(
    f"{'Method':28s}"
    f"{'IoU':>10s}"
    f"{'Dice':>10s}"
    f"{'Precision':>12s}"
    f"{'Recall':>10s}"
    f"{'Specificity':>13s}"
)

print("-" * 75)

for result in summary:

    print(
        f"{result['method']:28s}"
        f"{result['iou']*100:9.2f}%"
        f"{result['dice']*100:9.2f}%"
        f"{result['precision']*100:11.2f}%"
        f"{result['recall']*100:9.2f}%"
        f"{result['specificity']*100:12.2f}%"
    )


# ============================================================
# BEST METHOD
# ============================================================

best = summary[0]

print()
print("=" * 75)
print("BEST METHOD")
print("=" * 75)

print(f"Method     : {best['method']}")
print(f"IoU        : {best['iou']*100:.2f}%")
print(f"Dice       : {best['dice']*100:.2f}%")
print(f"Precision  : {best['precision']*100:.2f}%")
print(f"Recall     : {best['recall']*100:.2f}%")
print(f"Specificity: {best['specificity']*100:.2f}%")
print(f"Accuracy   : {best['accuracy']*100:.2f}%")


# ============================================================
# SAVE
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "postprocessing_results.json"
)

with open(json_path, "w") as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# Save per-image results too

per_image_path = os.path.join(
    OUTPUT_DIR,
    "postprocessing_per_image.json"
)

with open(per_image_path, "w") as f:
    json.dump(
        all_results,
        f,
        indent=2
    )


print()
print("=" * 75)
print("DONE")
print("=" * 75)

print(f"Results: {json_path}")
print(f"Per-image results: {per_image_path}")
