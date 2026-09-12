import os

import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import structural_similarity
from dataset import Sentinel2SRDataset
from model import EDSR


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = r"C:\Users\anku islam\Downloads\sentinel2_sr_dataset\content\dataset"

# IMPORTANT:
# This checkpoint was trained using:
# features = 64
# num_blocks = 8
CHECKPOINT = r"checkpoints\edsr_epoch_20.pth"

# Number of TEST images to evaluate
NUM_SAMPLES = 541

# Create results folder
os.makedirs("results", exist_ok=True)


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

num_samples = min(
    NUM_SAMPLES,
    len(dataset)
)

print("TEST dataset:")
print("Test samples:", len(dataset))
print("Evaluating:", num_samples, "samples")


# ============================================================
# CREATE MODEL
# IMPORTANT:
# These settings MUST exactly match the checkpoint
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
print("Model configuration:")
print("  Input channels :", 4)
print("  Output channels:", 4)
print("  Features       :", 64)
print("  Residual blocks:", 8)
print("  Scale          :", 4)


# ============================================================
# METRIC STORAGE
# ============================================================

edsr_mse = []
edsr_psnr = []
edsr_ssim = []

bicubic_mse = []
bicubic_psnr = []
bicubic_ssim = []


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for i in range(num_samples):

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        lr, hr = dataset[i]

        lr = lr.unsqueeze(0).to(device)
        hr = hr.unsqueeze(0).to(device)


        # ----------------------------------------------------
        # EDSR SUPER-RESOLUTION
        # ----------------------------------------------------

        sr = model(lr)

        sr = torch.clamp(
            sr,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # BICUBIC BASELINE
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr,
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
        # MSE
        # ----------------------------------------------------

        mse_sr = F.mse_loss(
            sr,
            hr
        ).item()

        mse_bicubic = F.mse_loss(
            bicubic,
            hr
        ).item()


        # ----------------------------------------------------
        # PSNR
        # ----------------------------------------------------

        if mse_sr > 0:
            psnr_sr = 10 * np.log10(
                1.0 / mse_sr
            )
        else:
            psnr_sr = float("inf")

        if mse_bicubic > 0:
            psnr_bicubic = 10 * np.log10(
                1.0 / mse_bicubic
            )
        else:
            psnr_bicubic = float("inf")


        # ----------------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------------

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

        hr_np = (
            hr.squeeze(0)
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # SSIM FOR 4 BANDS
        # ----------------------------------------------------

        sr_ssim_bands = []
        bicubic_ssim_bands = []

        for band in range(4):

            sr_band_ssim = structural_similarity(
                hr_np[band],
                sr_np[band],
                data_range=1.0
            )

            bicubic_band_ssim = structural_similarity(
                hr_np[band],
                bicubic_np[band],
                data_range=1.0
            )

            sr_ssim_bands.append(
                sr_band_ssim
            )

            bicubic_ssim_bands.append(
                bicubic_band_ssim
            )


        # Average SSIM over 4 bands

        ssim_sr = np.mean(
            sr_ssim_bands
        )

        ssim_bicubic = np.mean(
            bicubic_ssim_bands
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        edsr_mse.append(mse_sr)
        edsr_psnr.append(psnr_sr)
        edsr_ssim.append(ssim_sr)

        bicubic_mse.append(mse_bicubic)
        bicubic_psnr.append(psnr_bicubic)
        bicubic_ssim.append(ssim_bicubic)


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (i + 1) % 10 == 0:

            print(
                f"Evaluated {i + 1}/{num_samples}"
            )


# ============================================================
# FINAL AVERAGES
# ============================================================

avg_edsr_mse = np.mean(edsr_mse)
avg_edsr_psnr = np.mean(edsr_psnr)
avg_edsr_ssim = np.mean(edsr_ssim)

avg_bicubic_mse = np.mean(bicubic_mse)
avg_bicubic_psnr = np.mean(bicubic_psnr)
avg_bicubic_ssim = np.mean(bicubic_ssim)


# ============================================================
# IMPROVEMENT
# ============================================================

psnr_improvement = (
    avg_edsr_psnr -
    avg_bicubic_psnr
)

ssim_improvement = (
    avg_edsr_ssim -
    avg_bicubic_ssim
)

mse_reduction = (
    avg_bicubic_mse -
    avg_edsr_mse
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print("=" * 60)
print("FINAL TEST EVALUATION RESULTS")
print("=" * 60)

print()

print("EDSR:")
print(
    f"Average MSE  : {avg_edsr_mse:.6f}"
)

print(
    f"Average PSNR : {avg_edsr_psnr:.4f} dB"
)

print(
    f"Average SSIM : {avg_edsr_ssim:.4f}"
)

print()

print("BICUBIC BASELINE:")
print(
    f"Average MSE  : {avg_bicubic_mse:.6f}"
)

print(
    f"Average PSNR : {avg_bicubic_psnr:.4f} dB"
)

print(
    f"Average SSIM : {avg_bicubic_ssim:.4f}"
)


# ============================================================
# EDSR IMPROVEMENT
# ============================================================

print()

print("=" * 60)
print("EDSR IMPROVEMENT OVER BICUBIC")
print("=" * 60)

print(
    f"PSNR improvement : {psnr_improvement:.4f} dB"
)

print(
    f"SSIM improvement : {ssim_improvement:.4f}"
)

print(
    f"MSE reduction    : {mse_reduction:.6f}"
)


# ============================================================
# COMPLETED
# ============================================================

print()

print("=" * 60)
print("Evaluation completed!")
print("=" * 60)