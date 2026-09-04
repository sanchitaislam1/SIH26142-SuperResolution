import os
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import Affine
import matplotlib.pyplot as plt

from src.model import EDSR


# ============================================================
# SETTINGS
# ============================================================

INPUT_TIF = "data/sentinel2_4band.tif"
CHECKPOINT = "checkpoints/edsr_epoch_20.pth"

OUTPUT_TIF = "results/real_tif_super_resolved.tif"
OUTPUT_PNG = "results/real_tif_comparison.png"

SCALE = 4
PATCH_SIZE = 32

os.makedirs("results", exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD TIFF
# ============================================================

print("\nReading:", INPUT_TIF)

with rasterio.open(INPUT_TIF) as src:

    image = src.read().astype(np.float32)

    profile = src.profile.copy()

    transform = src.transform

    crs = src.crs

    print("Input bands:", src.count)
    print("Input size:", src.width, "x", src.height)
    print("CRS:", crs)


if image.shape[0] != 4:
    raise ValueError(
        f"Expected 4 bands, but found {image.shape[0]}."
    )


# ============================================================
# NORMALIZE SENTINEL-2 DATA
# ============================================================

print("\nPreparing bands...")

# Sentinel-2 reflectance is often stored as 0–10000.
if np.nanmax(image) > 1.5:
    image = image / 10000.0

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=1.0,
    neginf=0.0
)

image = np.clip(image, 0.0, 1.0)

print(
    "Data range:",
    float(image.min()),
    "to",
    float(image.max())
)


# ============================================================
# LOAD EDSR
# ============================================================

print("\nLoading EDSR model...")

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

# Handle checkpoints saved either as a state_dict
# or inside a dictionary.
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# TILED SUPER-RESOLUTION
# ============================================================

_, height, width = image.shape

output_height = height * SCALE
output_width = width * SCALE

output = np.zeros(
    (4, output_height, output_width),
    dtype=np.float32
)

count = np.zeros(
    (1, output_height, output_width),
    dtype=np.float32
)

total_patches = (
    int(np.ceil(height / PATCH_SIZE))
    * int(np.ceil(width / PATCH_SIZE))
)

patch_number = 0


print("\nRunning 4× super-resolution...")
print("Input:", width, "x", height)
print("Output:", output_width, "x", output_height)
print("Total patches:", total_patches)


with torch.no_grad():

    for y in range(0, height, PATCH_SIZE):

        for x in range(0, width, PATCH_SIZE):

            patch = image[
                :,
                y:min(y + PATCH_SIZE, height),
                x:min(x + PATCH_SIZE, width)
            ]

            ph = patch.shape[1]
            pw = patch.shape[2]

            tensor = torch.from_numpy(
                patch
            ).unsqueeze(0)

            # Pad boundary patches.
            pad_h = PATCH_SIZE - ph
            pad_w = PATCH_SIZE - pw

            if pad_h > 0 or pad_w > 0:

                tensor = F.pad(
                    tensor,
                    (0, pad_w, 0, pad_h),
                    mode="reflect"
                )

            tensor = tensor.to(device)

            sr = model(tensor)

            sr = torch.clamp(
                sr,
                0.0,
                1.0
            )

            sr = sr.squeeze(0).cpu().numpy()

            valid_h = ph * SCALE
            valid_w = pw * SCALE

            output_y = y * SCALE
            output_x = x * SCALE

            output[
                :,
                output_y:output_y + valid_h,
                output_x:output_x + valid_w
            ] += sr[
                :,
                :valid_h,
                :valid_w
            ]

            count[
                :,
                output_y:output_y + valid_h,
                output_x:output_x + valid_w
            ] += 1.0

            patch_number += 1

            print(
                f"Patch {patch_number}/{total_patches}",
                end="\r"
            )


# ============================================================
# AVERAGE OVERLAPPING AREAS
# ============================================================

output = output / np.maximum(
    count,
    1.0
)

output = np.clip(
    output,
    0.0,
    1.0
)

print("\n\nSuper-resolution completed!")


# ============================================================
# SAVE GEOTIFF
# ============================================================

print("\nSaving GeoTIFF...")

new_profile = profile.copy()

new_profile.update(
    driver="GTiff",
    height=output_height,
    width=output_width,
    count=4,
    dtype="float32",
    compress="deflate",
    predictor=3,

    # Because resolution becomes 4× finer.
    transform=transform * Affine.scale(
        1 / SCALE,
        1 / SCALE
    )
)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    **new_profile
) as dst:

    dst.write(output[0], 1)
    dst.write(output[1], 2)
    dst.write(output[2], 3)
    dst.write(output[3], 4)

    dst.set_band_description(
        1,
        "B04 Red"
    )

    dst.set_band_description(
        2,
        "B03 Green"
    )

    dst.set_band_description(
        3,
        "B02 Blue"
    )

    dst.set_band_description(
        4,
        "B08 NIR"
    )

print("Saved:", OUTPUT_TIF)


# ============================================================
# RGB PREVIEW
# ============================================================

def make_rgb(img):

    # Our band order is:
    # B04 = Red
    # B03 = Green
    # B02 = Blue

    rgb = np.stack(
        [
            img[0],
            img[1],
            img[2]
        ],
        axis=-1
    )

    low = np.percentile(rgb, 2)
    high = np.percentile(rgb, 98)

    if high > low:
        rgb = (
            rgb - low
        ) / (
            high - low
        )
    else:
        rgb = np.zeros_like(rgb)

    return np.clip(
        rgb,
        0.0,
        1.0
    )


input_rgb = make_rgb(image)

output_rgb = make_rgb(output)


# ============================================================
# SAVE COMPARISON IMAGE
# ============================================================

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(input_rgb)

plt.title(
    f"Original Sentinel-2\n{width} × {height}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(output_rgb)

plt.title(
    f"EDSR 4× Output\n{output_width} × {output_height}"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved preview:", OUTPUT_PNG)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n========================================")
print("          INFERENCE COMPLETE")
print("========================================")

print("Input TIFF :", INPUT_TIF)
print("Output TIFF:", OUTPUT_TIF)
print("Preview    :", OUTPUT_PNG)

print(
    f"Input size : {width} × {height}"
)

print(
    f"Output size: {output_width} × {output_height}"
)

print(
    "Scale      : 4×"
)

print(
    "Bands      : B04, B03, B02, B08"
)

print("========================================")