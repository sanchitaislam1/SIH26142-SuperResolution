import os
import sys
import numpy as np
import torch
import rasterio

# ------------------------------------------------------------
# Make src/model.py importable
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import EDSR


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

BAND_FILES = [
    "B04.tif",   # Red
    "B03.tif",   # Green
    "B02.tif",   # Blue
    "B08.tif"    # NIR
]

CHECKPOINT = os.path.join(
    BASE_DIR,
    "checkpoints",
    "edsr_epoch_20.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sentinel2_edsr_super_resolved.tif"
)

SCALE = 4
PATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Sentinel-2 GeoTIFF EDSR Inference")
print("=" * 60)

print("Using device:", device)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking input files...")

for filename in BAND_FILES:

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    print("Found:", filename)


if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(
        f"Missing checkpoint:\n{CHECKPOINT}"
    )

print("Found checkpoint: edsr_epoch_20.pth")


# ============================================================
# READ SENTINEL-2 BANDS
# ============================================================

print("\nReading Sentinel-2 bands...")

bands = []
profile = None
reference_transform = None
reference_crs = None

for filename in BAND_FILES:

    path = os.path.join(DATA_DIR, filename)

    with rasterio.open(path) as src:

        band = src.read(1).astype(np.float32)

        print(
            filename,
            "shape =", band.shape,
            "dtype =", band.dtype,
            "min =", np.nanmin(band),
            "max =", np.nanmax(band)
        )

        bands.append(band)

        # Use B04 as reference for geospatial metadata
        if profile is None:

            profile = src.profile.copy()
            reference_transform = src.transform
            reference_crs = src.crs

        else:

            # Make sure all bands have same dimensions
            if band.shape != bands[0].shape:

                raise ValueError(
                    "The four bands do not have identical dimensions."
                )


# ============================================================
# STACK INTO 4-CHANNEL IMAGE
# ============================================================

image = np.stack(bands, axis=0)

print("\nCombined input shape:", image.shape)

channels, height, width = image.shape


# ============================================================
# NORMALIZATION
# ============================================================

print("\nPreparing reflectance values...")

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# The downloaded Sentinel-2 GeoTIFF data is commonly stored
# using integer reflectance values.
#
# If values are already between 0 and 1, keep them unchanged.
# Otherwise scale the reflectance values to the 0-1 range.

if image.max() > 1.0:

    print(
        "Input values are larger than 1."
    )

    print(
        "Scaling reflectance by 10000."
    )

    image = image / 10000.0

else:

    print(
        "Input already appears to be normalized to 0-1."
    )

image = np.clip(
    image,
    0.0,
    1.0
)

print(
    "Normalized range:",
    image.min(),
    "to",
    image.max()
)


# ============================================================
# LOAD EDSR MODEL
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

model.load_state_dict(
    checkpoint
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# OUTPUT ARRAY
# ============================================================

sr_height = height * SCALE
sr_width = width * SCALE

sr_image = np.zeros(
    (
        4,
        sr_height,
        sr_width
    ),
    dtype=np.float32
)


# ============================================================
# PATCH-BASED SUPER RESOLUTION
# ============================================================

print("\nStarting super-resolution...")

print(
    f"Input resolution : {width} × {height}"
)

print(
    f"Output resolution: {sr_width} × {sr_height}"
)

total_patches_y = (height + PATCH_SIZE - 1) // PATCH_SIZE
total_patches_x = (width + PATCH_SIZE - 1) // PATCH_SIZE

total_patches = (
    total_patches_y *
    total_patches_x
)

patch_number = 0


with torch.no_grad():

    for y in range(
        0,
        height,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width,
            PATCH_SIZE
        ):

            patch_number += 1

            y_end = min(
                y + PATCH_SIZE,
                height
            )

            x_end = min(
                x + PATCH_SIZE,
                width
            )

            patch = image[
                :,
                y:y_end,
                x:x_end
            ]

            original_h = patch.shape[1]
            original_w = patch.shape[2]


            # ------------------------------------------------
            # Pad edge patches to 32 × 32
            # ------------------------------------------------

            padded = np.zeros(
                (
                    4,
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                dtype=np.float32
            )

            padded[
                :,
                :original_h,
                :original_w
            ] = patch


            # ------------------------------------------------
            # Convert to PyTorch tensor
            # ------------------------------------------------

            tensor = torch.from_numpy(
                padded
            ).unsqueeze(0)

            tensor = tensor.to(device)


            # ------------------------------------------------
            # EDSR
            # ------------------------------------------------

            output = model(tensor)

            output = torch.clamp(
                output,
                0.0,
                1.0
            )


            output = output.squeeze(
                0
            ).cpu().numpy()


            # ------------------------------------------------
            # Crop padded output
            # ------------------------------------------------

            output_h = original_h * SCALE
            output_w = original_w * SCALE

            output = output[
                :,
                :output_h,
                :output_w
            ]


            # ------------------------------------------------
            # Put output into final image
            # ------------------------------------------------

            sr_image[
                :,
                y * SCALE:y * SCALE + output_h,
                x * SCALE:x * SCALE + output_w
            ] = output


            if (
                patch_number == 1
                or
                patch_number % 10 == 0
                or
                patch_number == total_patches
            ):

                print(
                    f"Processed "
                    f"{patch_number}/{total_patches} patches"
                )


# ============================================================
# SAVE SUPER-RESOLVED GEOTIFF
# ============================================================

print("\nSaving super-resolved GeoTIFF...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Update GeoTIFF metadata
# ------------------------------------------------------------

output_profile = profile.copy()

output_profile.update(

    driver="GTiff",

    height=sr_height,

    width=sr_width,

    count=4,

    dtype="float32",

    compress="lzw"
)


# ------------------------------------------------------------
# IMPORTANT:
# Pixel size becomes 1/4 because resolution increased 4×.
# ------------------------------------------------------------

old_transform = reference_transform

new_transform = rasterio.Affine(
    old_transform.a / SCALE,
    old_transform.b,
    old_transform.c,
    old_transform.d,
    old_transform.e / SCALE,
    old_transform.f
)

output_profile.update(
    transform=new_transform,
    crs=reference_crs
)


# ============================================================
# WRITE OUTPUT
# ============================================================

with rasterio.open(
    OUTPUT_FILE,
    "w",
    **output_profile
) as dst:

    dst.write(
        sr_image.astype(np.float32)
    )

    # Add band descriptions
    dst.set_band_description(
        1,
        "B04 Red - EDSR"
    )

    dst.set_band_description(
        2,
        "B03 Green - EDSR"
    )

    dst.set_band_description(
        3,
        "B02 Blue - EDSR"
    )

    dst.set_band_description(
        4,
        "B08 NIR - EDSR"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("SUCCESS!")

print("=" * 60)

print(
    "Input:",
    f"{width} × {height}"
)

print(
    "Output:",
    f"{sr_width} × {sr_height}"
)

print(
    "Scale factor:",
    "4×"
)

print(
    "Bands:",
    "B04, B03, B02, B08"
)

print(
    "Output file:"
)

print(
    OUTPUT_FILE
)

print("=" * 60)