import os
import torch
import matplotlib.pyplot as plt

from dataset import Sentinel2SRDataset
from model import EDSR


# =========================
# SETTINGS
# =========================

DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"
CHECKPOINT = r"checkpoints\edsr_epoch_20.pth"

# Make sure results folder exists
os.makedirs("results", exist_ok=True)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================
# LOAD DATASET
# =========================

dataset = Sentinel2SRDataset(
    DATASET_ROOT,
    split="train"
)

lr, hr = dataset[0]

# Add batch dimension
lr = lr.unsqueeze(0).to(device)
hr = hr.unsqueeze(0).to(device)

print("LR input shape :", lr.shape)
print("HR target shape:", hr.shape)


# =========================
# LOAD MODEL
# =========================

model = EDSR(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
    scale=4
)

model = model.to(device)


# =========================
# LOAD TRAINED CHECKPOINT
# =========================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")


# =========================
# SUPER-RESOLUTION
# =========================

with torch.no_grad():
    sr = model(lr)

print("SR output shape:", sr.shape)


# =========================
# CLIP VALUES
# =========================

sr = torch.clamp(sr, 0.0, 1.0)


# =========================
# SAVE OUTPUT
# =========================

sr_cpu = sr.squeeze(0).cpu()

torch.save(
    sr_cpu,
    "results/super_resolved.pt"
)

print("Super-resolved image saved!")


# =========================
# VISUALIZATION
# =========================

lr_img = lr.squeeze(0).cpu()
hr_img = hr.squeeze(0).cpu()
sr_img = sr.squeeze(0).cpu()


# Use RGB channels
# Channel 0 = B04 (Red)
# Channel 1 = B03 (Green)
# Channel 2 = B02 (Blue)

lr_rgb = lr_img[[0, 1, 2]].permute(1, 2, 0)
hr_rgb = hr_img[[0, 1, 2]].permute(1, 2, 0)
sr_rgb = sr_img[[0, 1, 2]].permute(1, 2, 0)


# =========================
# PLOT COMPARISON
# =========================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(lr_rgb)
plt.title("LR Input (32x32)")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(sr_rgb)
plt.title("Super-Resolved (128x128)")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(hr_rgb)
plt.title("HR Ground Truth (128x128)")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "results/comparison.png",
    dpi=200
)

plt.show()

print("Comparison saved to: results/comparison.png")