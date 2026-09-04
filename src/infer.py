import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import EDSR


# =========================
# SETTINGS
# =========================

LR_FILE = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset\train\lr\patch_0218.npy"

CHECKPOINT = r"checkpoints\edsr_epoch_20.pth"

OUTPUT_FILE = "results/inference_output.npy"

os.makedirs("results", exist_ok=True)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# LOAD LR IMAGE
# =========================

lr = np.load(LR_FILE).astype(np.float32)

print("Original LR shape:", lr.shape)

lr = torch.from_numpy(lr)

# Add batch dimension
lr = lr.unsqueeze(0).to(device)

print("Model input shape:", lr.shape)


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

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# =========================
# SUPER-RESOLUTION
# =========================

with torch.no_grad():

    sr = model(lr)

    sr = torch.clamp(
        sr,
        0.0,
        1.0
    )

print("SR output shape:", sr.shape)


# =========================
# SAVE 4-BAND OUTPUT
# =========================

sr_cpu = sr.squeeze(0).cpu()

np.save(
    OUTPUT_FILE,
    sr_cpu.numpy()
)

print("Super-resolved image saved to:")
print(OUTPUT_FILE)


# =========================
# RGB VISUALIZATION
# =========================

sr_img = sr_cpu.numpy()

# B04 = Red
# B03 = Green
# B02 = Blue

sr_rgb = sr_img[[0, 1, 2]]

# CHW → HWC
sr_rgb = np.transpose(
    sr_rgb,
    (1, 2, 0)
)


# Normalize for display
sr_rgb = sr_rgb - sr_rgb.min()

if sr_rgb.max() > 0:
    sr_rgb = sr_rgb / sr_rgb.max()


plt.figure(figsize=(7, 7))

plt.imshow(sr_rgb)

plt.title("EDSR Super-Resolved Sentinel-2")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "results/inference_visualization.png",
    dpi=200
)

plt.show()

print("Visualization saved to:")
print("results/inference_visualization.png")