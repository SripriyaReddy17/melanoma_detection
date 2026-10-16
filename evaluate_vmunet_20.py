import os
import json
import cv2
import torch
import numpy as np
from PIL import Image

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet


# ============================================================
# Paths
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
    "outputs/sam2_batch/coarse_masks"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("UltraLight VM-UNet — 20 Image Batch")
print("=" * 60)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load validation split
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

validation_images = split["validation_images"][:20]

print("Validation images:", len(validation_images))


# ============================================================
# Load model
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

# Handle either raw state_dict or checkpoint dictionary
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint, strict=True)

model.to(device)
model.eval()

print("VM-UNet loaded successfully")


# ============================================================
# Helper
# ============================================================

def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return image, tensor, original_h, original_w


# ============================================================
# Process images
# ============================================================

metadata = []

with torch.no_grad():

    for index, filename in enumerate(validation_images, start=1):

        print()
        print(f"[{index}/20] {filename}")

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        image, tensor, original_h, original_w = load_image(
            image_path
        )

        tensor = tensor.to(device)

        # VM-UNet already applies sigmoid internally
        prediction = model(tensor)

        prediction = prediction.squeeze().detach().cpu().numpy()

        # Resize probability map back to original resolution
        probability = cv2.resize(
            prediction,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR
        )

        # IMPORTANT:
        # We previously found 0.1 to be the best validation threshold.
        binary_mask = (probability >= 0.1).astype(np.uint8)

        # Save binary mask as 0/255 PNG
        mask_name = filename.replace(".jpg", ".png")

        mask_path = os.path.join(
            OUTPUT_DIR,
            mask_name
        )

        cv2.imwrite(
            mask_path,
            binary_mask * 255
        )

        foreground_pixels = int(binary_mask.sum())
        total_pixels = binary_mask.size
        foreground_percent = (
            foreground_pixels / total_pixels * 100
        )

        print(
            f"  Size: {original_w}x{original_h}"
        )

        print(
            f"  Foreground: "
            f"{foreground_pixels:,} "
            f"({foreground_percent:.2f}%)"
        )

        print(
            f"  Saved: {mask_path}"
        )

        metadata.append({
            "image": filename,
            "image_path": image_path,
            "coarse_mask": mask_path,
            "height": original_h,
            "width": original_w,
            "foreground_pixels": foreground_pixels,
            "foreground_percent": foreground_percent
        })


# ============================================================
# Save metadata
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)


print()
print("=" * 60)
print("DONE")
print("=" * 60)
print("Images processed:", len(metadata))
print("Masks:", OUTPUT_DIR)
print("Metadata:", metadata_path)
