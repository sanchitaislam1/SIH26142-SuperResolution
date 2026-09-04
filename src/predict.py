import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataset import Sentinel2SRDataset
from model import EDSR


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"

# IMPORTANT:
# This is the same checkpoint used for the final test evaluation.
CHECKPOINT = r"checkpoints\edsr_epoch_20.pth"

# Test images to visualize
IMAGE_INDICES = [0, 50, 100, 200, 300, 400]

# Output folder
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)
print("=" * 60)


# ============================================================
# LOAD TEST DATASET
# ============================================================

dataset = Sentinel2SRDataset(
    DATASET_ROOT,
    split="test"
)

print("TEST dataset:")
print("Test samples:", len(dataset))

# Remove invalid indices
valid_indices = [
    i for i in IMAGE_INDICES
    if 0 <= i < len(dataset)
]

print("Images to visualize:", valid_indices)


# ============================================================
# CREATE MODEL
# MUST EXACTLY MATCH EPOCH 20 CHECKPOINT
# ============================================================

model = EDSR(
    in_channels=4,
    out_channels=4,
    features=64,
    num_blocks=8,
    scale=4
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully!")
print("Checkpoint:", CHECKPOINT)

print()
print("Model configuration:")
print("  Input channels : 4")
print("  Output channels: 4")
print("  Features       : 64")
print("  Residual blocks: 8")
print("  Scale          : 4")


# ============================================================
# FUNCTION: CREATE RGB IMAGE
# ============================================================

def create_rgb(image):
    """
    Convert 4-band Sentinel-2 image into RGB.

    Band 0 = Blue
    Band 1 = Green
    Band 2 = Red
    Band 3 = NIR

    RGB visualization:
    Red   = Band 2
    Green = Band 1
    Blue  = Band 0
    """

    rgb = np.stack(
        [
            image[2],
            image[1],
            image[0]
        ],
        axis=-1
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    return rgb


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    for image_index in valid_indices:

        print()
        print("-" * 60)
        print("Processing test image:", image_index)
        print("-" * 60)

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        lr, hr = dataset[image_index]

        lr_input = lr.unsqueeze(0).to(device)


        # ----------------------------------------------------
        # EDSR SUPER-RESOLUTION
        # ----------------------------------------------------

        sr = model(lr_input)

        sr = torch.clamp(
            sr,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # BICUBIC UPSAMPLING
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr_input,
            size=(128, 128),
            mode="bicubic",
            align_corners=False
        )

        bicubic = torch.clamp(
            bicubic,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

        lr_np = lr.numpy()

        sr_np = (
            sr.squeeze(0)
            .cpu()
            .numpy()
        )

        bicubic_np = (
            bicubic.squeeze(0)
            .cpu()
            .numpy()
        )

        hr_np = hr.numpy()


        # ----------------------------------------------------
        # CREATE RGB IMAGES
        # ----------------------------------------------------

        lr_rgb = create_rgb(lr_np)

        bicubic_rgb = create_rgb(
            bicubic_np
        )

        sr_rgb = create_rgb(
            sr_np
        )

        hr_rgb = create_rgb(
            hr_np
        )


        # ====================================================
        # CREATE COMPARISON FIGURE
        # ====================================================

        plt.figure(
            figsize=(16, 4)
        )


        # ----------------------------------------------------
        # LR INPUT
        # ----------------------------------------------------

        plt.subplot(1, 4, 1)

        plt.imshow(lr_rgb)

        plt.title(
            "LR Input\n32 × 32"
        )

        plt.axis("off")


        # ----------------------------------------------------
        # BICUBIC
        # ----------------------------------------------------

        plt.subplot(1, 4, 2)

        plt.imshow(bicubic_rgb)

        plt.title(
            "Bicubic\n128 × 128"
        )

        plt.axis("off")


        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        plt.subplot(1, 4, 3)

        plt.imshow(sr_rgb)

        plt.title(
            "EDSR\n128 × 128"
        )

        plt.axis("off")


        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        plt.subplot(1, 4, 4)

        plt.imshow(hr_rgb)

        plt.title(
            "Ground Truth\n128 × 128"
        )

        plt.axis("off")


        # ====================================================
        # SAVE COMPARISON
        # ====================================================

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_test_{image_index}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


        # ====================================================
        # SAVE EDSR 4-BAND OUTPUT
        # ====================================================

        npy_path = os.path.join(
            OUTPUT_DIR,
            f"edsr_output_test_{image_index}.npy"
        )

        np.save(
            npy_path,
            sr_np
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "Comparison saved to:",
            output_path
        )

        print(
            "EDSR output saved to:",
            npy_path
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("ALL PREDICTIONS COMPLETED!")
print("=" * 60)

print()
print("Generated comparisons:")

for image_index in valid_indices:
    print(
        f"results/comparison_test_{image_index}.png"
    )

print()
print("Generated EDSR outputs:")

for image_index in valid_indices:
    print(
        f"results/edsr_output_test_{image_index}.npy"
    )

print()
print("=" * 60)