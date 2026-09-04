import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window

# ============================================================
# SETTINGS
# ============================================================

SCALE = 4
PATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your existing trained checkpoint
CHECKPOINT = "checkpoints/edsr_epoch_20.pth"


# ============================================================
# IMPORT YOUR EXISTING EDSR MODEL
# ============================================================

from src.model import EDSR


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = EDSR(
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale=4
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(np.float32)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Sentinel-2 reflectance is commonly represented
    # in scaled integer form or normalized floating point.
    # Keep already-normalized data unchanged.
    if image.max() > 1.0:
        image = image / 10000.0

    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# READ SENTINEL-2 GEOTIFF
# ============================================================

def read_sentinel2_geotiff(path):

    with rasterio.open(path) as src:

        print("\nInput GeoTIFF:")
        print(path)

        print("Number of bands:", src.count)
        print("Width:", src.width)
        print("Height:", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)

        if src.count < 4:
            raise ValueError(
                "GeoTIFF must contain at least 4 bands."
            )

        # Expected order:
        # B04 = Red
        # B03 = Green
        # B02 = Blue
        # B08 = NIR
        #
        # IMPORTANT:
        # This assumes the uploaded GeoTIFF already has
        # these four bands in this order.

        image = src.read(
            [1, 2, 3, 4]
        ).astype(np.float32)

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

    image = normalize_image(image)

    return image, profile, transform, crs


# ============================================================
# PAD IMAGE
# ============================================================

def pad_image(image):

    bands, height, width = image.shape

    padded_height = (
        (height + PATCH_SIZE - 1)
        // PATCH_SIZE
    ) * PATCH_SIZE

    padded_width = (
        (width + PATCH_SIZE - 1)
        // PATCH_SIZE
    ) * PATCH_SIZE

    padded = np.zeros(
        (
            bands,
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    padded[:, :height, :width] = image

    return padded


# ============================================================
# EDSR INFERENCE
# ============================================================

def super_resolve(image, model):

    padded = pad_image(image)

    bands, height, width = padded.shape

    output_height = height * SCALE
    output_width = width * SCALE

    output = np.zeros(
        (
            bands,
            output_height,
            output_width
        ),
        dtype=np.float32
    )

    print("\nRunning EDSR inference...")
    print(
        "Input:",
        padded.shape
    )

    patch_count = 0

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

            patch = padded[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            tensor = torch.from_numpy(
                patch
            ).unsqueeze(0).to(DEVICE)

            with torch.inference_mode():

                sr = model(tensor)

            sr = (
                sr.squeeze(0)
                .cpu()
                .numpy()
            )

            output[
                :,
                y * SCALE:(y + PATCH_SIZE) * SCALE,
                x * SCALE:(x + PATCH_SIZE) * SCALE
            ] = sr

            patch_count += 1

            if patch_count % 10 == 0:
                print(
                    f"Processed {patch_count} patches..."
                )

    output = np.clip(
        output,
        0.0,
        1.0
    )

    print(
        "\nSuper-resolution completed."
    )

    print(
        "Output:",
        output.shape
    )

    return output


# ============================================================
# SAVE SUPER-RESOLVED GEOTIFF
# ============================================================

def save_geotiff(
    output,
    profile,
    transform,
    crs,
    output_path
):

    bands, height, width = output.shape

    new_profile = profile.copy()

    new_profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": bands,
            "dtype": "float32",
            "transform": transform,
            "crs": crs,
            "compress": "deflate"
        }
    )

    with rasterio.open(
        output_path,
        "w",
        **new_profile
    ) as dst:

        dst.write(
            output.astype(np.float32)
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SENTINEL-2 EDSR GEOTIFF INFERENCE")
    print("=" * 60)

    # --------------------------------------------------------
    # INPUT FILE
    # --------------------------------------------------------

    input_path = input(
        "\nEnter Sentinel-2 GeoTIFF path: "
    ).strip()

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"File not found: {input_path}"
        )

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_path = input(
        "Enter output GeoTIFF path: "
    ).strip()

    if not output_path:

        output_path = (
            "results/"
            "sentinel2_super_resolved.tif"
        )

    os.makedirs(
        os.path.dirname(output_path)
        if os.path.dirname(output_path)
        else ".",
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\nLoading EDSR model..."
    )

    model = load_model()

    print(
        "Model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image, profile, transform, crs = (
        read_sentinel2_geotiff(
            input_path
        )
    )

    # --------------------------------------------------------
    # SUPER RESOLUTION
    # --------------------------------------------------------

    output = super_resolve(
        image,
        model
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_geotiff(
        output,
        profile,
        transform,
        crs,
        output_path
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()