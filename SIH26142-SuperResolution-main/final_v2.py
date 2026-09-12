import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from src.model import EDSR


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentinel-2 Super Resolution",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = os.path.join(
    "checkpoints",
    "edsr_epoch_20.pth"
)

METRICS_FILE = os.path.join(
    "results",
    "metrics_541.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCALE = 4
PATCH_SIZE = 32


# ============================================================
# LOAD EDSR MODEL
# ============================================================

@st.cache_resource
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

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# DATA PREPROCESSING
# ============================================================

def prepare_bands(data):

    data = data.astype(np.float32)

    max_value = np.nanmax(data)

    # Sentinel-2 reflectance scaling
    if max_value > 1.5:
        data = data / 10000.0

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return np.clip(
        data,
        0.0,
        1.0
    )


# ============================================================
# RGB VISUALIZATION
# ============================================================

def create_rgb(image):

    # Expected order:
    # B04 = Red
    # B03 = Green
    # B02 = Blue
    # B08 = NIR

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2]
        ],
        axis=-1
    )

    low, high = np.percentile(
        rgb,
        2
    ), np.percentile(
        rgb,
        98
    )

    if high > low:
        rgb = (
            (rgb - low)
            / (high - low)
        )
    else:
        rgb = np.zeros_like(rgb)

    return np.clip(
        rgb,
        0.0,
        1.0
    )


# ============================================================
# SINGLE BAND VISUALIZATION
# ============================================================

def display_band(band):

    low, high = np.percentile(
        band,
        2
    ), np.percentile(
        band,
        98
    )

    if high > low:
        band = (
            (band - low)
            / (high - low)
        )
    else:
        band = np.zeros_like(band)

    return np.clip(
        band,
        0.0,
        1.0
    )


# ============================================================
# BICUBIC BASELINE
# ============================================================

def run_bicubic_inference(image):

    """
    Upscale the 4-band Sentinel-2 image
    using bicubic interpolation.

    This acts as the conventional baseline
    for comparison with the EDSR model.
    """

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        bicubic = F.interpolate(
            tensor,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )

    bicubic = torch.clamp(
        bicubic,
        0.0,
        1.0
    )

    return (
        bicubic
        .squeeze(0)
        .cpu()
        .numpy()
    )


# ============================================================
# EDSR TILED INFERENCE
# ============================================================

def run_tiled_inference(model, image):

    _, height, width = image.shape

    out_h = height * SCALE
    out_w = width * SCALE

    output = np.zeros(
        (
            4,
            out_h,
            out_w
        ),
        dtype=np.float32
    )

    count = np.zeros(
        (
            1,
            out_h,
            out_w
        ),
        dtype=np.float32
    )

    total = (
        int(np.ceil(height / PATCH_SIZE))
        *
        int(np.ceil(width / PATCH_SIZE))
    )

    done = 0

    progress = st.progress(0.0)
    status = st.empty()

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

                # --------------------------------------------
                # Extract patch
                # --------------------------------------------

                patch = image[
                    :,
                    y:min(
                        y + PATCH_SIZE,
                        height
                    ),
                    x:min(
                        x + PATCH_SIZE,
                        width
                    )
                ]

                ph = patch.shape[1]
                pw = patch.shape[2]

                tensor = torch.from_numpy(
                    patch
                ).unsqueeze(0)

                # --------------------------------------------
                # Reflect padding for boundary patches
                # --------------------------------------------

                pad_h = PATCH_SIZE - ph
                pad_w = PATCH_SIZE - pw

                if pad_h or pad_w:

                    tensor = F.pad(
                        tensor,
                        (
                            0,
                            pad_w,
                            0,
                            pad_h
                        ),
                        mode="reflect"
                    )

                tensor = tensor.to(DEVICE)

                # --------------------------------------------
                # EDSR inference
                # --------------------------------------------

                with torch.inference_mode():
                    sr = model(tensor)
                    sr = torch.clamp(
                        sr,
                        0.0,
                        1.0
                    )

                # Move the result to CPU before releasing the
                # temporary inference tensor.
                sr = (
                    sr
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                del tensor

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                # --------------------------------------------
                # Output coordinates
                # --------------------------------------------

                valid_h = ph * SCALE
                valid_w = pw * SCALE

                oy = y * SCALE
                ox = x * SCALE

                # --------------------------------------------
                # Stitch patches together
                # --------------------------------------------

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

                progress.progress(
                    done / total
                )

                status.text(
                    f"Processing patch {done}/{total}..."
                )

    progress.empty()
    status.empty()

    output = (
        output
        /
        np.maximum(
            count,
            1.0
        )
    )

    return np.clip(
        output,
        0.0,
        1.0
    )


# ============================================================
# CREATE SUPER-RESOLVED GEOTIFF
# ============================================================

def create_geotiff_bytes(
    sr,
    source_profile
):

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
            *
            rasterio.Affine.scale(
                1 / SCALE,
                1 / SCALE
            )
        )
    )

    with MemoryFile() as memfile:

        with memfile.open(
            **profile
        ) as dst:

            # B04 — Red
            dst.write(
                sr[0],
                1
            )

            # B03 — Green
            dst.write(
                sr[1],
                2
            )

            # B02 — Blue
            dst.write(
                sr[2],
                3
            )

            # B08 — NIR
            dst.write(
                sr[3],
                4
            )

            band_names = [
                "B04 Red",
                "B03 Green",
                "B02 Blue",
                "B08 NIR"
            ]

            for i, name in enumerate(
                band_names,
                1
            ):

                dst.set_band_description(
                    i,
                    name
                )

        return memfile.read()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        f"Could not load the model: {e}"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛰️ Sentinel-2 Super-Resolution"
)

st.markdown(
    """
### AI-Based Satellite Image Enhancement

Upload a **4-band Sentinel-2 GeoTIFF** containing
**B04, B03, B02 and B08**.

The trained **EDSR deep-learning model**
enhances the spatial resolution by **4×**.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Model Information"
    )

    st.write(
        "**Model:** EDSR"
    )

    st.write(
        "**Input:** 4-band Sentinel-2 GeoTIFF"
    )

    st.write(
        "**Scale Factor:** 4×"
    )

    st.write(
        "**Residual Blocks:** 8"
    )

    st.write(
        "**Features:** 64"
    )

    st.divider()

    st.write(
        "**Bands used:**"
    )

    st.write(
        "B04 — Red"
    )

    st.write(
        "B03 — Green"
    )

    st.write(
        "B02 — Blue"
    )

    st.write(
        "B08 — NIR"
    )

    st.divider()

    if DEVICE.type == "cuda":

        st.success(
            "GPU available"
        )

    else:

        st.info(
            "Running on CPU"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "📂 Upload Sentinel-2 GeoTIFF"
)

uploaded_file = st.file_uploader(
    "Upload a 4-band Sentinel-2 .tif / .tiff file",
    type=["tif", "tiff"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ GEOTIFF
    # --------------------------------------------------------

    try:

        with MemoryFile(
            uploaded_file.getvalue()
        ) as memfile:

            with memfile.open() as src:

                # Check band count
                if src.count != 4:

                    st.error(
                        f"""
This GeoTIFF contains {src.count} bands.

Exactly 4 bands are required:

B04, B03, B02, B08
"""
                    )

                    st.stop()

                # Read image
                lr_raw = src.read()

                # Save source profile
                source_profile = (
                    src.profile.copy()
                )

                # Save CRS
                input_crs = src.crs

    except Exception as e:

        st.error(
            f"Unable to read the GeoTIFF: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    lr_np = prepare_bands(
        lr_raw
    )

    _, height, width = (
        lr_np.shape
    )

    out_h = height * SCALE
    out_w = width * SCALE


    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    st.success(
        "Valid 4-band Sentinel-2 GeoTIFF detected."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Input Size",
        f"{width} × {height}"
    )

    c2.metric(
        "Output Size",
        f"{out_w} × {out_h}"
    )

    c3.metric(
        "Scale",
        "4×"
    )

    if input_crs:

        st.caption(
            f"CRS: {input_crs}"
        )

    else:

        st.caption(
            "CRS information is not present in this file."
        )


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    lr_rgb = create_rgb(
        lr_np
    )

    st.subheader(
        "🖼️ Input Image"
    )

    st.image(
        lr_rgb,
        caption=(
            f"Original Sentinel-2 — "
            f"{width} × {height}"
        ),
        width="stretch"
    )


    # ========================================================
    # GENERATE SUPER RESOLUTION
    # ========================================================

    if st.button(
        "🚀 Generate 4× Super-Resolution",
        type="primary",
        width="stretch"
    ):

        # ----------------------------------------------------
        # BICUBIC BASELINE
        # ----------------------------------------------------

        with st.spinner(
            "Running Bicubic baseline..."
        ):

            bicubic_np = (
                run_bicubic_inference(
                    lr_np
                )
            )

            bicubic_rgb = create_rgb(
                bicubic_np
            )


        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        with st.spinner(
            "Running EDSR super-resolution..."
        ):

            sr_np = run_tiled_inference(
                model,
                lr_np
            )

            sr_rgb = create_rgb(
                sr_np
            )

            # Create GeoTIFF
            output_bytes = (
                create_geotiff_bytes(
                    sr_np,
                    source_profile
                )
            )

            # Save local result
            os.makedirs(
                "results",
                exist_ok=True
            )

            output_path = os.path.join(
                "results",
                "streamlit_super_resolved.tif"
            )

            with open(
                output_path,
                "wb"
            ) as f:

                f.write(
                    output_bytes
                )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        st.success(
            "Super-resolution completed successfully!"
        )

        st.divider()


        # ====================================================
        # VISUAL COMPARISON
        # ====================================================

        st.header(
            "🔍 Super-Resolution Comparison"
        )

        st.caption(
            "Comparison of the original image, "
            "conventional Bicubic interpolation, "
            "and the EDSR deep-learning output."
        )

        c1, c2, c3 = st.columns(3)


        # Original
        with c1:

            st.subheader(
                "Original"
            )

            st.image(
                lr_rgb,
                caption=(
                    f"Original — "
                    f"{width} × {height}"
                ),
                width="stretch"
            )


        # Bicubic
        with c2:

            st.subheader(
                "Bicubic"
            )

            st.image(
                bicubic_rgb,
                caption=(
                    f"Bicubic — "
                    f"{out_w} × {out_h}"
                ),
                width="stretch"
            )


        # EDSR
        with c3:

            st.subheader(
                "EDSR"
            )

            st.image(
                sr_rgb,
                caption=(
                    f"EDSR — "
                    f"{out_w} × {out_h}"
                ),
                width="stretch"
            )


        # ====================================================
        # SPECTRAL BAND VISUALIZATION
        # ====================================================

        st.divider()

        st.header(
            "🛰️ Spectral Band Visualization"
        )

        names = [
            "B04 — Red",
            "B03 — Green",
            "B02 — Blue",
            "B08 — NIR"
        ]

        cols = st.columns(4)

        for i in range(4):

            with cols[i]:

                st.subheader(
                    names[i]
                )

                st.image(
                    display_band(
                        sr_np[i]
                    ),
                    caption=(
                        f"EDSR {names[i]}"
                    ),
                    width="stretch"
                )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.header(
            "📈 Model Performance"
        )

        if os.path.exists(
            METRICS_FILE
        ):

            try:

                df = pd.read_csv(
                    METRICS_FILE
                )

                # Average metrics
                ep = df[
                    "edsr_psnr"
                ].mean()

                bp = df[
                    "bicubic_psnr"
                ].mean()

                es = df[
                    "edsr_ssim"
                ].mean()

                bs = df[
                    "bicubic_ssim"
                ].mean()

                em = df[
                    "edsr_mse"
                ].mean()

                bm = df[
                    "bicubic_mse"
                ].mean()


                # --------------------------------------------
                # Metric cards
                # --------------------------------------------

                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Average PSNR",
                        f"{ep:.2f} dB",
                        f"+{ep - bp:.2f} dB vs Bicubic"
                    )


                with c2:

                    st.metric(
                        "Average SSIM",
                        f"{es:.4f}",
                        f"+{es - bs:.4f} vs Bicubic"
                    )


                with c3:

                    st.metric(
                        "Average MSE",
                        f"{em:.6f}",
                        f"{bm - em:.6f} lower"
                    )


                # --------------------------------------------
                # Comparison table
                # --------------------------------------------

                table = pd.DataFrame(
                    {
                        "Metric": [
                            "PSNR (dB)",
                            "SSIM",
                            "MSE"
                        ],

                        "Bicubic": [
                            bp,
                            bs,
                            bm
                        ],

                        "EDSR": [
                            ep,
                            es,
                            em
                        ]
                    }
                )


                st.dataframe(
                    table,
                    hide_index=True,
                    width="stretch"
                )


                st.caption(
                    "Benchmark results are calculated "
                    "on the 541-image test dataset. "
                    "They are not metrics for the "
                    "currently uploaded image."
                )


            except Exception as e:

                st.warning(
                    f"Unable to load benchmark metrics: {e}"
                )

        else:

            st.warning(
                "Benchmark metrics file not found."
            )


        # ====================================================
        # DOWNLOAD GEOTIFF
        # ====================================================

        st.divider()

        st.header(
            "💾 Download Super-Resolved GeoTIFF"
        )

        st.download_button(
            "⬇️ Download 4-Band Super-Resolved GeoTIFF",
            data=output_bytes,
            file_name="sentinel2_super_resolved.tif",
            mime="image/tiff",
            width="stretch"
        )


        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        st.divider()

        st.header(
            "📊 Output Information"
        )

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Bands",
                "4"
            )


        with c2:

            st.metric(
                "Resolution",
                f"{out_w} × {out_h}"
            )


        with c3:

            st.metric(
                "Scale Factor",
                "4×"
            )


        with c4:

            st.metric(
                "Output Format",
                ".tif"
            )


        st.caption(
            "The output contains B04, B03, B02 "
            "and B08 bands and retains the source "
            "GeoTIFF's geographic reference."
        )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        """
👆 Upload a 4-band Sentinel-2 `.tif` / `.tiff` file to begin.

**Required band order:** B04, B03, B02, B08

The model will generate a **4× super-resolved GeoTIFF**.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentinel-2 Super-Resolution | "
    "EDSR Deep Learning Model"
)