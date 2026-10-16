import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT = Path("/mnt/d/pytest_cache")

IMAGE_PATH = (
    PROJECT
    / "datasets"
    / "ISIC2018"
    / "ISIC2018_Task1-2_Training_Input"
    / "ISIC_0000001.jpg"
)

MODEL_CODE = (
    PROJECT
    / "models"
    / "segmentation"
    / "ultralight_vmunet"
)

WEIGHTS = (
    PROJECT
    / "models"
    / "segmentation"
    / "UltraLight_VM_UNet.pth"
)

OUTPUT_DIR = PROJECT / "outputs" / "sam2_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Import VM-UNet
# ---------------------------------------------------------
sys.path.insert(0, str(MODEL_CODE))

from UltraLight_VM_UNet import UltraLight_VM_UNet


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("VM-UNET COARSE MASK GENERATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------
image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("Image:", IMAGE_PATH)
print("Original shape:", image_rgb.shape)


# ---------------------------------------------------------
# Load VM-UNet
# ---------------------------------------------------------
model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8, 16, 24, 32, 48, 64]
)

checkpoint = torch.load(
    WEIGHTS,
    map_location=device
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

checkpoint = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in checkpoint.items()
}

model.load_state_dict(checkpoint, strict=True)

model.to(device)
model.eval()

print("VM-UNet loaded successfully.")


# ---------------------------------------------------------
# Prepare input
# ---------------------------------------------------------
resized = cv2.resize(
    image_rgb,
    (224, 224),
    interpolation=cv2.INTER_LINEAR
)

tensor = torch.from_numpy(resized).float()
tensor = tensor.permute(2, 0, 1)
tensor = tensor.unsqueeze(0)
tensor = tensor / 255.0
tensor = tensor.to(device)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
with torch.no_grad():
    output = model(tensor)

# IMPORTANT:
# VM-UNet already applies sigmoid internally.
probability = output.squeeze().cpu().numpy()

# Validated threshold from our VM-UNet evaluation
small_mask = (probability >= 0.1).astype(np.uint8)


# ---------------------------------------------------------
# Restore original resolution
# ---------------------------------------------------------
mask = cv2.resize(
    small_mask,
    (image_rgb.shape[1], image_rgb.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

print("Mask foreground pixels:", int(mask.sum()))
print(
    "Mask percentage:",
    round(100 * mask.mean(), 2),
    "%"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
mask_path = OUTPUT_DIR / "vmunet_coarse_mask.png"
image_path = OUTPUT_DIR / "original_ISIC_0000001.jpg"

cv2.imwrite(
    str(mask_path),
    mask * 255
)

cv2.imwrite(
    str(image_path),
    image_bgr
)

print("\nSaved:")
print("Original:", image_path)
print("VM-UNet mask:", mask_path)

print("\n" + "=" * 60)
print("VM-UNET MASK READY FOR SAM2")
print("=" * 60)
