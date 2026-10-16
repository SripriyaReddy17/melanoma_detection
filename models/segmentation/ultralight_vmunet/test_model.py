import torch
import sys
from pathlib import Path

from UltraLight_VM_UNet import UltraLight_VM_UNet

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "segmentation" / "UltraLight_VM_UNet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8,16,24,32,48,64]
)

print("\nModel created successfully.")

checkpoint = torch.load(MODEL_PATH, map_location=device)

print("Checkpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Checkpoint keys:")
    print(list(checkpoint.keys())[:20])

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Remove possible DataParallel prefix
state_dict = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}

missing, unexpected = model.load_state_dict(state_dict, strict=True)

print("\nMissing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("First missing keys:", missing[:10])

if unexpected:
    print("First unexpected keys:", unexpected[:10])

model = model.to(device)
model.eval()

# Dummy inference
x = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    output = model(x)

print("\nInput shape :", x.shape)

if isinstance(output, (tuple, list)):
    print("Output is:", type(output))
    for i, out in enumerate(output):
        print(f"Output {i} shape:", out.shape)
else:
    print("Output shape:", output.shape)

print("\nMODEL TEST COMPLETED")
