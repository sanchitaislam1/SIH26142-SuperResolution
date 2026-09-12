import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
import rasterio
from rasterio.io import MemoryFile
from src.model import EDSR


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = os.path.join("checkpoints", "edsr_epoch_20.pth")
METRICS_FILE = os.path.join("results", "metrics_541.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE = 4

# Smaller patches use less memory on Render CPU.
PATCH_SIZE = 32


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

    model.load_state_dict(checkpoint)

    return model.to(DEVICE).eval()


try:
    model = load_model()
    MODEL_STATUS = f"Model loaded successfully • Device: {DEVICE}"
except Exception as e:
    model = None
    MODEL_STATUS = f"Model loading failed: {e}"


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_bands(data):
    """
    Convert Sentinel-2 bands to float32 in the range 0-1.
    """

    data = data.astype(np.float32)

    if np.nanmax(data) > 1.5:
        data = data / 10000.0

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return np.clip(data, 0.0, 1.0)


def create_rgb(image):
    """
    Create an RGB visualization from:
    B04 = Red
    B03 = Green
    B02 = Blue
    """

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2]
        ],
        axis=-1
    )

    low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)

    if high > low:
        rgb = (rgb - low) / (high - low)
    else:
        rgb = np.zeros_like(rgb)

    return np.clip(rgb, 0.0, 1.0)


def display_band(band):
    """
    Normalize an individual spectral band for display.
    """

    low, high = np.percentile(band, 2), np.percentile(band, 98)

    if high > low:
        band = (band - low) / (high - low)
    else:
        band = np.zeros_like(band)

    return np.clip(band, 0.0, 1.0)


# ============================================================
# EDSR TILED INFERENCE
# ============================================================

def run_tiled_inference(model, image, progress=gr.Progress()):
    """
    Run EDSR on the image tile-by-tile.

    This avoids sending the entire Sentinel-2 image
    through the neural network at once.
    """

    _, height, width = image.shape

    out_h = height * SCALE
    out_w = width * SCALE

    output = np.zeros(
        (4, out_h, out_w),
        dtype=np.float32
    )

    count = np.zeros(
        (1, out_h, out_w),
        dtype=np.float32
    )

    total_y = int(np.ceil(height / PATCH_SIZE))
    total_x = int(np.ceil(width / PATCH_SIZE))
    total = total_y * total_x

    done = 0

    with torch.inference_mode():

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

                pad_h = PATCH_SIZE - ph
                pad_w = PATCH_SIZE - pw

                # Replicate padding is safer for very small
                # edge patches than reflect padding.
                if pad_h or pad_w:
                    tensor = F.pad(
                        tensor,
                        (0, pad_w, 0, pad_h),
                        mode="replicate"
                    )

                tensor = tensor.to(DEVICE)

                sr = model(tensor)

                sr = torch.clamp(
                    sr,
                    0.0,
                    1.0
                )

                sr = sr.squeeze(0).cpu().numpy()

                valid_h = ph * SCALE
                valid_w = pw * SCALE

                oy = y * SCALE
                ox = x * SCALE

                output[
                    :,
                    oy:oy + valid_h,
                    ox:ox + valid_w
                ] += sr[
                    :,
                    :valid_h,
                    :valid_w
                ]

                count[
                    :,
                    oy:oy + valid_h,
                    ox:ox + valid_w
                ] += 1.0

                done += 1

                progress(
                    done / total,
                    desc=f"EDSR processing patch {done}/{total}"
                )

                del tensor
                del sr

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    output = output / np.maximum(
        count,
        1.0
    )

    return np.clip(
        output,
        0.0,
        1.0
    )


# ============================================================
# CREATE OUTPUT GEOTIFF
# ============================================================

def create_geotiff_bytes(sr, source_profile):

    profile = source_profile.copy()

    profile.update(
        driver="GTiff",
        height=sr.shape[1],
        width=sr.shape[2],
        count=4,
        dtype="float32",
        compress="deflate",
        predictor=3,
        transform=(
            profile["transform"]
            * rasterio.Affine.scale(
                1 / SCALE,
                1 / SCALE
            )
        )
    )

    with MemoryFile() as memfile:

        with memfile.open(**profile) as dst:

            names = [
                "B04 Red",
                "B03 Green",
                "B02 Blue",
                "B08 NIR"
            ]

            for i, name in enumerate(names, 1):

                dst.write(
                    sr[i - 1],
                    i
                )

                dst.set_band_description(
                    i,
                    name
                )

        return memfile.read()


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_geotiff(
    file_path,
    progress=gr.Progress()
):

    if file_path is None:
        raise gr.Error(
            "Please upload a 4-band Sentinel-2 GeoTIFF."
        )

    if model is None:
        raise gr.Error(
            f"EDSR model could not be loaded. {MODEL_STATUS}"
        )

    try:

        # ----------------------------------------------------
        # READ INPUT TIFF
        # ----------------------------------------------------

        progress(
            0.02,
            desc="Reading Sentinel-2 GeoTIFF..."
        )

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        with MemoryFile(raw_bytes) as memfile:

            with memfile.open() as src:

                if src.count != 4:

                    raise gr.Error(
                        f"This GeoTIFF contains {src.count} bands. "
                        "Exactly 4 bands are required: "
                        "B04, B03, B02, B08."
                    )

                lr_raw = src.read()

                source_profile = src.profile.copy()

                input_crs = src.crs

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        progress(
            0.05,
            desc="Preparing input bands..."
        )

        lr_np = prepare_bands(lr_raw)

        _, height, width = lr_np.shape

        out_h = height * SCALE
        out_w = width * SCALE

        # Original RGB preview
        lr_rgb = create_rgb(lr_np)

        # ----------------------------------------------------
        # SKIP EXPENSIVE FULL-IMAGE BICUBIC
        # ----------------------------------------------------
        #
        # The previous version performed:
        #
        # run_bicubic_inference(lr_np)
        #
        # on all 4 bands and the entire image.
        #
        # On Render CPU this could take several minutes
        # while the UI remained at 5%.
        #
        # We therefore use the original RGB preview here
        # instead of performing the expensive baseline.
        #
        # EDSR remains the actual super-resolution operation.
        # ----------------------------------------------------

        progress(
            0.08,
            desc="Starting EDSR super-resolution..."
        )

        bicubic_rgb = lr_rgb

        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        sr_np = run_tiled_inference(
            model,
            lr_np,
            progress
        )

        sr_rgb = create_rgb(sr_np)

        # ----------------------------------------------------
        # CREATE OUTPUT GEOTIFF
        # ----------------------------------------------------

        progress(
            0.95,
            desc="Creating 4× GeoTIFF..."
        )

        output_bytes = create_geotiff_bytes(
            sr_np,
            source_profile
        )

        os.makedirs(
            "results",
            exist_ok=True
        )

        output_path = os.path.abspath(
            os.path.join(
                "results",
                "gradio_super_resolved.tif"
            )
        )

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(output_bytes)

        # ----------------------------------------------------
        # LOAD BENCHMARK METRICS
        # ----------------------------------------------------

        if os.path.exists(METRICS_FILE):

            try:

                df = pd.read_csv(
                    METRICS_FILE
                )

                ep = df["edsr_psnr"].mean()
                bp = df["bicubic_psnr"].mean()

                es = df["edsr_ssim"].mean()
                bs = df["bicubic_ssim"].mean()

                em = df["edsr_mse"].mean()
                bm = df["bicubic_mse"].mean()

                metrics = (
                    f"**Average PSNR:** "
                    f"{ep:.2f} dB "
                    f"(+{ep - bp:.2f} dB vs Bicubic)\n\n"

                    f"**Average SSIM:** "
                    f"{es:.4f} "
                    f"(+{es - bs:.4f} vs Bicubic)\n\n"

                    f"**Average MSE:** "
                    f"{em:.6f} "
                    f"({bm - em:.6f} lower than Bicubic)\n\n"

                    "Benchmark: 541-image test dataset; "
                    "not metrics for the uploaded image."
                )

            except Exception as e:

                metrics = (
                    f"Unable to load benchmark metrics: {e}"
                )

        else:

            metrics = (
                "Benchmark metrics file not found."
            )

        # ----------------------------------------------------
        # OUTPUT INFORMATION
        # ----------------------------------------------------

        info = (
            f"**Input:** {width} × {height}\n\n"
            f"**Output:** {out_w} × {out_h}\n\n"
            f"**Scale:** 4×\n\n"
            f"**Bands:** B04, B03, B02, B08\n\n"
            f"**CRS:** "
            f"{input_crs if input_crs else 'Not present'}"
        )

        progress(
            1.0,
            desc="Completed!"
        )

        return (
            lr_rgb,
            bicubic_rgb,
            sr_rgb,
            display_band(sr_np[0]),
            display_band(sr_np[1]),
            display_band(sr_np[2]),
            display_band(sr_np[3]),
            metrics,
            info,
            output_path
        )

    except gr.Error:
        raise

    except Exception as e:

        raise gr.Error(
            f"Unable to process the GeoTIFF: {e}"
        )


# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title="Sentinel-2 Super-Resolution"
) as demo:

    gr.Markdown(
        "# 🛰️ Sentinel-2 Super-Resolution"
    )

    gr.Markdown(
        """
### AI-Based Satellite Image Enhancement

Upload a **4-band Sentinel-2 GeoTIFF** containing:

- **B04 — Red**
- **B03 — Green**
- **B02 — Blue**
- **B08 — NIR**

The trained **EDSR deep-learning model** enhances
spatial resolution by **4×**.
"""
    )

    gr.Markdown(
        "**Model:** EDSR • "
        "**Residual Blocks:** 8 • "
        "**Features:** 64 • "
        "**Scale:** 4×"
    )

    gr.Markdown(
        f"**Status:** {MODEL_STATUS}"
    )

    # --------------------------------------------------------
    # UPLOAD + BUTTON
    # --------------------------------------------------------

    with gr.Row():

        with gr.Column():

            input_file = gr.File(
                label="📂 Upload Sentinel-2 GeoTIFF",
                file_types=[".tif", ".tiff"],
                type="filepath"
            )

            process_button = gr.Button(
                "🚀 Generate 4× Super-Resolution",
                variant="primary"
            )

        with gr.Column():

            output_info = gr.Markdown(
                "### 📊 Output Information\n\n"
                "No image processed yet."
            )

    # --------------------------------------------------------
    # IMAGE COMPARISON
    # --------------------------------------------------------

    gr.Markdown(
        "## 🔍 Super-Resolution Comparison"
    )

    gr.Markdown(
        "The middle panel is shown as a lightweight "
        "input baseline to avoid expensive full-image "
        "Bicubic processing on CPU."
    )

    with gr.Row():

        original_output = gr.Image(
            label="Original Sentinel-2",
            type="numpy"
        )

        bicubic_output = gr.Image(
            label="Input Baseline",
            type="numpy"
        )

        edsr_output = gr.Image(
            label="EDSR 4×",
            type="numpy"
        )

    # --------------------------------------------------------
    # SPECTRAL BANDS
    # --------------------------------------------------------

    gr.Markdown(
        "## 🛰️ Spectral Band Visualization"
    )

    with gr.Row():

        b04_output = gr.Image(
            label="B04 — Red",
            type="numpy"
        )

        b03_output = gr.Image(
            label="B03 — Green",
            type="numpy"
        )

        b02_output = gr.Image(
            label="B02 — Blue",
            type="numpy"
        )

        b08_output = gr.Image(
            label="B08 — NIR",
            type="numpy"
        )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    gr.Markdown(
        "## 📈 Model Performance"
    )

    metrics_output = gr.Markdown(
        "Benchmark metrics will appear after processing."
    )

    download_output = gr.File(
        label="💾 Download Super-Resolved GeoTIFF"
    )

    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    process_button.click(
        process_geotiff,
        inputs=input_file,
        outputs=[
            original_output,
            bicubic_output,
            edsr_output,
            b04_output,
            b03_output,
            b02_output,
            b08_output,
            metrics_output,
            output_info,
            download_output
        ],
        show_progress="full"
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )
