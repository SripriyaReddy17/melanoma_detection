import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet

IMAGE_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
MODEL_PATH = "/mnt/d/pytest_cache/models/segmentation/UltraLight_VM_UNet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8,16,24,32,48,64]
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

state_dict = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}

model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])

print("Device:", DEVICE)
print("Images:", len(files))

with torch.no_grad():

    for filename in files[:10]:

        image_id = os.path.splitext(filename)[0]

        image_path = os.path.join(IMAGE_DIR, filename)
        mask_path = os.path.join(
            MASK_DIR,
            image_id + "_segmentation.png"
        )

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        x = transform(image).unsqueeze(0).to(DEVICE)

        output = model(x)

        probability = torch.sigmoid(output)
        probability = probability.squeeze().cpu().numpy()

        prediction = probability >= 0.5

        mask = mask.resize((224,224), Image.NEAREST)
        target = np.array(mask) > 0

        pred_pixels = prediction.sum()
        target_pixels = target.sum()
        total_pixels = prediction.size

        print("\n", filename)
        print("--------------------------------")
        print("Raw output:")
        print(" min :", float(output.min()))
        print(" max :", float(output.max()))
        print(" mean:", float(output.mean()))

        print("Probability:")
        print(" min :", float(probability.min()))
        print(" max :", float(probability.max()))
        print(" mean:", float(probability.mean()))

        print("Prediction foreground:",
              pred_pixels,
              "/",
              total_pixels,
              f"({pred_pixels/total_pixels*100:.2f}%)")

        print("Ground truth foreground:",
              target_pixels,
              "/",
              total_pixels,
              f"({target_pixels/total_pixels*100:.2f}%)")

print("\nDEBUG COMPLETED")
