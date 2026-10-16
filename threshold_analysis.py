import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
MODEL_PATH = "/mnt/d/pytest_cache/models/segmentation/UltraLight_VM_UNet.pth"


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


# Remove DataParallel prefix
state_dict = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model = model.to(DEVICE)
model.eval()


# ============================================================
# TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# METRICS
# ============================================================

def calculate_dice(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + 1e-7
    )

    return dice


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
# THRESHOLDS
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
# EVALUATION
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
            continue

        # Load image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Load mask
        mask = Image.open(
            mask_path
        ).convert("L")

        # Image tensor
        image_tensor = image_transform(
            image
        ).unsqueeze(0).to(DEVICE)

        # Resize mask
        mask = mask.resize(
            (224, 224),
            Image.NEAREST
        )

        mask = np.array(mask)

        # Ground truth
        target = mask > 0

        # Model prediction
        output = model(image_tensor)

        # IMPORTANT:
        # This checkpoint already produces
        # probability-like output.
        prediction = output.squeeze().cpu().numpy()

        # Test every threshold
        for threshold in thresholds:

            pred = prediction >= threshold

            dice = calculate_dice(
                pred,
                target
            )

            threshold_dice[
                float(threshold)
            ].append(dice)

        # Progress
        if (idx + 1) % 100 == 0:

            print(
                f"[{idx + 1}/{len(image_files)}]"
            )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 65)
print("THRESHOLD ANALYSIS")
print("=" * 65)

best_threshold = None
best_dice = -1

for threshold in thresholds:

    mean_dice = np.mean(
        threshold_dice[
            float(threshold)
        ]
    )

    print(
        f"Threshold {threshold:.1f}"
        f" -> Mean Dice = {mean_dice:.4f}"
        f" ({mean_dice * 100:.2f}%)"
    )

    if mean_dice > best_dice:

        best_dice = mean_dice
        best_threshold = threshold


print()
print("=" * 65)
print("BEST THRESHOLD")
print("=" * 65)

print(
    f"Best threshold : {best_threshold:.1f}"
)

print(
    f"Best Dice      : {best_dice:.4f}"
    f" ({best_dice * 100:.2f}%)"
)

print("=" * 65)
