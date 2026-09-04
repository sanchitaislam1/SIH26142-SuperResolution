import os
import csv
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
CHECKPOINT = r"checkpoints\edsr_epoch_20.pth"

OUTPUT_DIR = "results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metrics_541.csv")

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

num_samples = len(dataset)

print("TEST dataset:")
print("Test samples:", num_samples)


# ============================================================
# CREATE MODEL
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
print("  Input channels :", 4)
print("  Output channels:", 4)
print("  Features       :", 64)
print("  Residual blocks:", 8)
print("  Scale          :", 4)


# ============================================================
# METRIC STORAGE
# ============================================================

results = []

edsr_mse_all = []
edsr_psnr_all = []
edsr_ssim_all = []

bicubic_mse_all = []
bicubic_psnr_all = []
bicubic_ssim_all = []


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
        # EDSR
        # ----------------------------------------------------

        sr = model(lr)

        sr = torch.clamp(
            sr,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # BICUBIC
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

        mse_edsr = F.mse_loss(
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

        if mse_edsr > 0:
            psnr_edsr = 10 * np.log10(
                1.0 / mse_edsr
            )
        else:
            psnr_edsr = float("inf")


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
        # SSIM
        # ----------------------------------------------------

        edsr_ssim_bands = []
        bicubic_ssim_bands = []

        for band in range(4):

            ssim_edsr_band = structural_similarity(
                hr_np[band],
                sr_np[band],
                data_range=1.0
            )

            ssim_bicubic_band = structural_similarity(
                hr_np[band],
                bicubic_np[band],
                data_range=1.0
            )

            edsr_ssim_bands.append(
                ssim_edsr_band
            )

            bicubic_ssim_bands.append(
                ssim_bicubic_band
            )


        ssim_edsr = np.mean(
            edsr_ssim_bands
        )

        ssim_bicubic = np.mean(
            bicubic_ssim_bands
        )


        # ----------------------------------------------------
        # IMPROVEMENT
        # ----------------------------------------------------

        psnr_improvement = (
            psnr_edsr -
            psnr_bicubic
        )

        ssim_improvement = (
            ssim_edsr -
            ssim_bicubic
        )

        mse_reduction = (
            mse_bicubic -
            mse_edsr
        )


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        edsr_mse_all.append(mse_edsr)
        edsr_psnr_all.append(psnr_edsr)
        edsr_ssim_all.append(ssim_edsr)

        bicubic_mse_all.append(mse_bicubic)
        bicubic_psnr_all.append(psnr_bicubic)
        bicubic_ssim_all.append(ssim_bicubic)


        results.append([
            i,
            mse_edsr,
            psnr_edsr,
            ssim_edsr,
            mse_bicubic,
            psnr_bicubic,
            ssim_bicubic,
            psnr_improvement,
            ssim_improvement,
            mse_reduction
        ])


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (i + 1) % 10 == 0:

            print(
                f"Evaluated {i + 1}/{num_samples}"
            )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_index",
        "edsr_mse",
        "edsr_psnr",
        "edsr_ssim",
        "bicubic_mse",
        "bicubic_psnr",
        "bicubic_ssim",
        "psnr_improvement",
        "ssim_improvement",
        "mse_reduction"
    ])

    writer.writerows(results)


# ============================================================
# FINAL AVERAGES
# ============================================================

avg_edsr_mse = np.mean(edsr_mse_all)
avg_edsr_psnr = np.mean(edsr_psnr_all)
avg_edsr_ssim = np.mean(edsr_ssim_all)

avg_bicubic_mse = np.mean(bicubic_mse_all)
avg_bicubic_psnr = np.mean(bicubic_psnr_all)
avg_bicubic_ssim = np.mean(bicubic_ssim_all)


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
print("FINAL TEST METRICS")
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

print("BICUBIC:")
print(
    f"Average MSE  : {avg_bicubic_mse:.6f}"
)
print(
    f"Average PSNR : {avg_bicubic_psnr:.4f} dB"
)
print(
    f"Average SSIM : {avg_bicubic_ssim:.4f}"
)

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

print()

print("=" * 60)
print("METRICS CSV SAVED")
print("=" * 60)

print(
    "File:",
    OUTPUT_CSV
)

print()
print("Evaluation completed!")
print("=" * 60)