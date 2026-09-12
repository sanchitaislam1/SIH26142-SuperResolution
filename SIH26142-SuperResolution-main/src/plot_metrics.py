import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "results/metrics_541.csv"
OUTPUT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

print("=" * 60)
print("METRICS ANALYSIS")
print("=" * 60)

print("Total test images:", len(df))
print()

# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

df["psnr_improvement"] = (
    df["edsr_psnr"] - df["bicubic_psnr"]
)

df["ssim_improvement"] = (
    df["edsr_ssim"] - df["bicubic_ssim"]
)

df["mse_reduction"] = (
    df["bicubic_mse"] - df["edsr_mse"]
)

# ============================================================
# GRAPH 1: PSNR COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["edsr_psnr"],
    label="EDSR"
)

plt.plot(
    df["image_index"],
    df["bicubic_psnr"],
    label="Bicubic"
)

plt.xlabel("Test Image Index")
plt.ylabel("PSNR (dB)")
plt.title("EDSR vs Bicubic - PSNR Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "psnr_comparison.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# GRAPH 2: SSIM COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["edsr_ssim"],
    label="EDSR"
)

plt.plot(
    df["image_index"],
    df["bicubic_ssim"],
    label="Bicubic"
)

plt.xlabel("Test Image Index")
plt.ylabel("SSIM")
plt.title("EDSR vs Bicubic - SSIM Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "ssim_comparison.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# GRAPH 3: MSE COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["edsr_mse"],
    label="EDSR"
)

plt.plot(
    df["image_index"],
    df["bicubic_mse"],
    label="Bicubic"
)

plt.xlabel("Test Image Index")
plt.ylabel("MSE")
plt.title("EDSR vs Bicubic - MSE Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "mse_comparison.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# GRAPH 4: PSNR IMPROVEMENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["psnr_improvement"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Test Image Index")
plt.ylabel("PSNR Improvement (dB)")
plt.title("PSNR Improvement of EDSR over Bicubic")

plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "psnr_improvement.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# GRAPH 5: SSIM IMPROVEMENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["ssim_improvement"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Test Image Index")
plt.ylabel("SSIM Improvement")
plt.title("SSIM Improvement of EDSR over Bicubic")

plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "ssim_improvement.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# GRAPH 6: MSE REDUCTION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["image_index"],
    df["mse_reduction"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Test Image Index")
plt.ylabel("MSE Reduction")
plt.title("MSE Reduction of EDSR over Bicubic")

plt.grid(True)

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "mse_reduction.png"
)

plt.savefig(path, dpi=200)
plt.close()

print("Saved:", path)

# ============================================================
# FINAL SUMMARY
# ============================================================

average_edsr_psnr = df["edsr_psnr"].mean()
average_bicubic_psnr = df["bicubic_psnr"].mean()

average_edsr_ssim = df["edsr_ssim"].mean()
average_bicubic_ssim = df["bicubic_ssim"].mean()

average_edsr_mse = df["edsr_mse"].mean()
average_bicubic_mse = df["bicubic_mse"].mean()

average_psnr_gain = df["psnr_improvement"].mean()
average_ssim_gain = df["ssim_improvement"].mean()
average_mse_reduction = df["mse_reduction"].mean()

print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Average EDSR PSNR    : {average_edsr_psnr:.4f} dB")
print(f"Average Bicubic PSNR : {average_bicubic_psnr:.4f} dB")
print(f"Average PSNR Gain    : {average_psnr_gain:.4f} dB")

print()

print(f"Average EDSR SSIM    : {average_edsr_ssim:.4f}")
print(f"Average Bicubic SSIM : {average_bicubic_ssim:.4f}")
print(f"Average SSIM Gain    : {average_ssim_gain:.4f}")

print()

print(f"Average EDSR MSE     : {average_edsr_mse:.6f}")
print(f"Average Bicubic MSE  : {average_bicubic_mse:.6f}")
print(f"Average MSE Reduction: {average_mse_reduction:.6f}")

print()
print("=" * 60)
print("ALL GRAPHS GENERATED SUCCESSFULLY!")
print("=" * 60)