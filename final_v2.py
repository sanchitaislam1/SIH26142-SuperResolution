import os
import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from src.model import EDSR


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentinel-2 Super-Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT CONFIGURATION
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

BAND_NAMES = [
    "B04 — Red",
    "B03 — Green",
    "B02 — Blue",
    "B08 — NIR",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        color: #9aa7b8;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 750;
        margin-top: 1rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-bottom: 1rem;
    }

    .success-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(50,205,120,0.25);
        background: rgba(50,205,120,0.08);
        margin-bottom: 1rem;
    }

    .small-text {
        color: #9aa7b8;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = EDSR(
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale=4,
    )

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif (
            "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_bands(data):
    """
    Convert Sentinel-2 data to float32 reflectance-like
    values in the [0, 1] range.
    """

    data = data.astype(np.float32)

    finite_values = data[np.isfinite(data)]

    if finite_values.size == 0:
        raise ValueError(
            "The uploaded image contains no valid pixel values."
        )

    max_value = np.max(finite_values)

    # Sentinel-2 commonly uses scaled reflectance values
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
# DISPLAY HELPERS
# ============================================================

def percentile_stretch(image, low_percentile=2, high_percentile=98):
    """
    Percentile stretch for visualization.
    Does not modify the actual model output.
    """

    image = image.astype(np.float32)

    low = np.percentile(
        image,
        low_percentile
    )

    high = np.percentile(
        image,
        high_percentile
    )

    if high > low:
        image = (
            image - low
        ) / (
            high - low
        )

    else:
        image = np.zeros_like(image)

    return np.clip(
        image,
        0.0,
        1.0
    )


def create_rgb(image):
    """
    RGB:
    B04 = Red
    B03 = Green
    B02 = Blue
    """

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2],
        ],
        axis=-1
    )

    return percentile_stretch(rgb)


def create_false_color(image):
    """
    False-color composite:
    B08 NIR = Red
    B04 Red = Green
    B03 Green = Blue
    """

    false_color = np.stack(
        [
            image[3],
            image[0],
            image[1],
        ],
        axis=-1
    )

    return percentile_stretch(false_color)


def display_band(band):
    return percentile_stretch(
        band
    )


def calculate_ndvi(image):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    """

    red = image[0]
    nir = image[3]

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir),
        where=denominator != 0
    )

    return np.clip(
        ndvi,
        -1.0,
        1.0
    )


def normalize_ndvi_for_display(ndvi):
    """
    Convert NDVI [-1, 1] into [0, 1] for display.
    """

    return (
        ndvi + 1.0
    ) / 2.0


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

def run_bicubic(image):
    """
    Conventional bicubic interpolation baseline.
    """

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    with torch.inference_mode():

        output = F.interpolate(
            tensor,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )

    output = torch.clamp(
        output,
        0.0,
        1.0
    )

    return output.squeeze(0).numpy()


# ============================================================
# TILED EDSR INFERENCE
# ============================================================

def run_tiled_inference(model, image):

    _, height, width = image.shape

    output_height = height * SCALE
    output_width = width * SCALE

    output = np.zeros(
        (
            4,
            output_height,
            output_width
        ),
        dtype=np.float32
    )

    count = np.zeros(
        (
            1,
            output_height,
            output_width
        ),
        dtype=np.float32
    )

    rows = int(
        np.ceil(
            height / PATCH_SIZE
        )
    )

    cols = int(
        np.ceil(
            width / PATCH_SIZE
        )
    )

    total_patches = rows * cols
    completed = 0

    progress = st.progress(
        0.0
    )

    status = st.empty()

    with torch.inference_mode():

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

                patch_height = patch.shape[1]
                patch_width = patch.shape[2]

                tensor = torch.from_numpy(
                    patch
                ).unsqueeze(0)

                pad_height = (
                    PATCH_SIZE
                    - patch_height
                )

                pad_width = (
                    PATCH_SIZE
                    - patch_width
                )

                if pad_height > 0 or pad_width > 0:

                    # Reflect padding can fail for very small
                    # dimensions, so use replicate in that case.
                    if (
                        patch_height > 1
                        and patch_width > 1
                    ):
                        tensor = F.pad(
                            tensor,
                            (
                                0,
                                pad_width,
                                0,
                                pad_height
                            ),
                            mode="reflect"
                        )
                    else:
                        tensor = F.pad(
                            tensor,
                            (
                                0,
                                pad_width,
                                0,
                                pad_height
                            ),
                            mode="replicate"
                        )

                tensor = tensor.to(
                    DEVICE
                )

                sr = model(
                    tensor
                )

                sr = torch.clamp(
                    sr,
                    0.0,
                    1.0
                )

                sr = sr.squeeze(
                    0
                ).cpu().numpy()

                valid_height = (
                    patch_height * SCALE
                )

                valid_width = (
                    patch_width * SCALE
                )

                output_y = y * SCALE
                output_x = x * SCALE

                output[
                    :,
                    output_y:
                    output_y + valid_height,
                    output_x:
                    output_x + valid_width
                ] += sr[
                    :,
                    :valid_height,
                    :valid_width
                ]

                count[
                    :,
                    output_y:
                    output_y + valid_height,
                    output_x:
                    output_x + valid_width
                ] += 1.0

                completed += 1

                progress.progress(
                    completed / total_patches
                )

                status.text(
                    f"Processing patch "
                    f"{completed}/{total_patches}"
                )

    progress.empty()
    status.empty()

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
# GEOTIFF CREATION
# ============================================================

def create_geotiff_bytes(
    sr,
    source_profile
):

    profile = source_profile.copy()

    original_transform = profile.get(
        "transform"
    )

    profile.update(
        driver="GTiff",
        height=sr.shape[1],
        width=sr.shape[2],
        count=4,
        dtype="float32",
        compress="deflate",
        predictor=3,
    )

    if original_transform is not None:

        profile["transform"] = (
            original_transform
            * Affine.scale(
                1 / SCALE,
                1 / SCALE
            )
        )

    # Remove values that may become incompatible
    # after changing the output data type.
    profile.pop(
        "nodata",
        None
    )

    with MemoryFile() as memfile:

        with memfile.open(
            **profile
        ) as dst:

            for i in range(4):

                dst.write(
                    sr[i].astype(
                        np.float32
                    ),
                    i + 1
                )

            for i, name in enumerate(
                BAND_NAMES,
                start=1
            ):

                dst.set_band_description(
                    i,
                    name
                )

        return memfile.read()


# ============================================================
# MODEL BENCHMARK
# ============================================================

def load_benchmark_metrics():

    if not os.path.exists(
        METRICS_FILE
    ):
        return None

    df = pd.read_csv(
        METRICS_FILE
    )

    required_columns = [
        "edsr_psnr",
        "bicubic_psnr",
        "edsr_ssim",
        "bicubic_ssim",
        "edsr_mse",
        "bicubic_mse",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing metric columns: "
            + ", ".join(missing)
        )

    return {
        "edsr_psnr": df["edsr_psnr"].mean(),
        "bicubic_psnr": df["bicubic_psnr"].mean(),

        "edsr_ssim": df["edsr_ssim"].mean(),
        "bicubic_ssim": df["bicubic_ssim"].mean(),

        "edsr_mse": df["edsr_mse"].mean(),
        "bicubic_mse": df["bicubic_mse"].mean(),

        "count": len(df),
    }


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    model_loaded = True

except Exception as error:

    model_loaded = False

    st.error(
        "❌ Unable to load the EDSR model."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Model Information"
    )

    st.markdown(
        "**Model**  \n"
        "EDSR"
    )

    st.markdown(
        "**Input**  \n"
        "4-band Sentinel-2"
    )

    st.markdown(
        "**Scale Factor**  \n"
        "4×"
    )

    st.markdown(
        "**Residual Blocks**  \n"
        "8"
    )

    st.markdown(
        "**Features**  \n"
        "64"
    )

    st.divider()

    st.markdown(
        "### 🛰️ Spectral Bands"
    )

    st.write(
        "🔴 B04 — Red"
    )

    st.write(
        "🟢 B03 — Green"
    )

    st.write(
        "🔵 B02 — Blue"
    )

    st.write(
        "🟣 B08 — NIR"
    )

    st.divider()

    st.markdown(
        "### 💻 Hardware"
    )

    if DEVICE.type == "cuda":

        st.success(
            "GPU available"
        )

    else:

        st.info(
            "Running on CPU"
        )

    st.divider()

    st.caption(
        f"Checkpoint: {os.path.basename(CHECKPOINT)}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ Sentinel-2 Super-Resolution</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based enhancement of multispectral Sentinel-2 imagery '
    'using a trained EDSR deep-learning model.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(
    "ℹ️ How the system works",
    expanded=False
):

    st.markdown(
        """
        **Input → Processing → Output**

        1. Upload a 4-band Sentinel-2 GeoTIFF.
        2. The application validates the input.
        3. The four spectral bands are normalized.
        4. The trained EDSR model processes the image in tiles.
        5. A 4× super-resolved image is reconstructed.
        6. Bicubic interpolation is generated as a conventional baseline.
        7. RGB, false-color, NDVI and individual spectral bands are displayed.
        8. The final 4-band GeoTIFF can be downloaded.

        **Important:** the benchmark PSNR, SSIM and MSE shown below
        come from the project's 541-image test dataset. They are
        not calculated from the currently uploaded image.
        """
    )


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 📂 Upload Sentinel-2 GeoTIFF"
)

uploaded_file = st.file_uploader(
    "Upload a 4-band Sentinel-2 .tif / .tiff file",
    type=[
        "tif",
        "tiff"
    ]
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        """
        👆 **Upload a 4-band Sentinel-2 GeoTIFF to begin.**

        Required band order:

        **B04 → B03 → B02 → B08**

        The system will generate a **4× super-resolved
        multispectral GeoTIFF**.
        """
    )

    st.divider()

    st.markdown(
        "### Expected pipeline"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Input",
        "4 Bands"
    )

    c2.metric(
        "Model",
        "EDSR"
    )

    c3.metric(
        "Scale",
        "4×"
    )

    c4.metric(
        "Output",
        "GeoTIFF"
    )


# ============================================================
# FILE PROCESSING
# ============================================================

else:

    # --------------------------------------------------------
    # READ GEOTIFF
    # --------------------------------------------------------

    try:

        file_bytes = uploaded_file.getvalue()

        with MemoryFile(
            file_bytes
        ) as memfile:

            with memfile.open() as src:

                band_count = src.count

                if band_count != 4:

                    st.error(
                        f"""
                        ❌ This GeoTIFF contains
                        **{band_count} bands**.

                        Exactly **4 bands** are required:

                        B04, B03, B02 and B08.
                        """
                    )

                    st.stop()

                lr_raw = src.read()

                source_profile = (
                    src.profile.copy()
                )

                input_crs = src.crs

                input_transform = (
                    src.transform
                )

                input_bounds = (
                    src.bounds
                )

                input_dtype = (
                    src.dtypes[0]
                )

                input_nodata = (
                    src.nodata
                )

    except Exception as error:

        st.error(
            "❌ Unable to read the GeoTIFF."
        )

        st.code(
            str(error)
        )

        st.stop()


    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    try:

        lr_np = prepare_bands(
            lr_raw
        )

    except Exception as error:

        st.error(
            "❌ Unable to prepare the image."
        )

        st.code(
            str(error)
        )

        st.stop()


    _, height, width = lr_np.shape

    output_height = (
        height * SCALE
    )

    output_width = (
        width * SCALE
    )


    # --------------------------------------------------------
    # VALIDATION MESSAGE
    # --------------------------------------------------------

    st.success(
        "✅ Valid 4-band Sentinel-2 GeoTIFF detected."
    )


    # --------------------------------------------------------
    # IMAGE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "## 📋 Input Information"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Input Size",
        f"{width} × {height}"
    )

    c2.metric(
        "Output Size",
        f"{output_width} × {output_height}"
    )

    c3.metric(
        "Bands",
        "4"
    )

    c4.metric(
        "Scale",
        "4×"
    )


    # --------------------------------------------------------
    # GEOSPATIAL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🌍 Geospatial Information",
        expanded=False
    ):

        geo1, geo2 = st.columns(2)

        with geo1:

            st.write(
                "**CRS:**",
                str(input_crs)
                if input_crs
                else "Not available"
            )

            st.write(
                "**Data type:**",
                input_dtype
            )

            st.write(
                "**NoData value:**",
                str(input_nodata)
                if input_nodata is not None
                else "Not specified"
            )

        with geo2:

            st.write(
                "**Bounds:**",
                str(input_bounds)
            )

            st.write(
                "**Original transform:**",
                str(input_transform)
            )


    # --------------------------------------------------------
    # INPUT RGB
    # --------------------------------------------------------

    lr_rgb = create_rgb(
        lr_np
    )


    st.divider()

    st.markdown(
        "## 🖼️ Input Image"
    )

    st.image(
        lr_rgb,
        caption=(
            f"Original Sentinel-2 RGB "
            f"— {width} × {height}"
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------------

    st.divider()

    generate = st.button(
        "🚀 Generate 4× Super-Resolution",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # RUN MODEL
    # ========================================================

    if generate:

        try:

            with st.spinner(
                "Running EDSR super-resolution..."
            ):

                # ----------------------------
                # EDSR
                # ----------------------------

                sr_np = run_tiled_inference(
                    model,
                    lr_np
                )

                # ----------------------------
                # Bicubic
                # ----------------------------

                bicubic_np = run_bicubic(
                    lr_np
                )

                # ----------------------------
                # Visualizations
                # ----------------------------

                sr_rgb = create_rgb(
                    sr_np
                )

                bicubic_rgb = create_rgb(
                    bicubic_np
                )

                sr_false_color = (
                    create_false_color(
                        sr_np
                    )
                )

                bicubic_false_color = (
                    create_false_color(
                        bicubic_np
                    )
                )

                sr_ndvi = calculate_ndvi(
                    sr_np
                )

                bicubic_ndvi = calculate_ndvi(
                    bicubic_np
                )

                # ----------------------------
                # GeoTIFF
                # ----------------------------

                output_bytes = (
                    create_geotiff_bytes(
                        sr_np,
                        source_profile
                    )
                )

                # ----------------------------
                # Save local copy
                # ----------------------------

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
                ) as output_file:

                    output_file.write(
                        output_bytes
                    )


            st.session_state[
                "sr_np"
            ] = sr_np

            st.session_state[
                "bicubic_np"
            ] = bicubic_np

            st.session_state[
                "sr_rgb"
            ] = sr_rgb

            st.session_state[
                "bicubic_rgb"
            ] = bicubic_rgb

            st.session_state[
                "sr_false_color"
            ] = sr_false_color

            st.session_state[
                "bicubic_false_color"
            ] = bicubic_false_color

            st.session_state[
                "sr_ndvi"
            ] = sr_ndvi

            st.session_state[
                "bicubic_ndvi"
            ] = bicubic_ndvi

            st.session_state[
                "output_bytes"
            ] = output_bytes


            st.success(
                "✅ Super-resolution completed successfully!"
            )

        except Exception as error:

            st.error(
                "❌ Super-resolution failed."
            )

            st.code(
                str(error)
            )

            st.stop()


    # ========================================================
    # DISPLAY RESULTS AFTER PROCESSING
    # ========================================================

    if "sr_np" in st.session_state:

        sr_np = st.session_state[
            "sr_np"
        ]

        bicubic_np = st.session_state[
            "bicubic_np"
        ]

        sr_rgb = st.session_state[
            "sr_rgb"
        ]

        bicubic_rgb = st.session_state[
            "bicubic_rgb"
        ]

        sr_false_color = (
            st.session_state[
                "sr_false_color"
            ]
        )

        bicubic_false_color = (
            st.session_state[
                "bicubic_false_color"
            ]
        )

        sr_ndvi = st.session_state[
            "sr_ndvi"
        ]

        bicubic_ndvi = (
            st.session_state[
                "bicubic_ndvi"
            ]
        )

        output_bytes = (
            st.session_state[
                "output_bytes"
            ]
        )


        # ====================================================
        # COMPARISON
        # ====================================================

        st.divider()

        st.markdown(
            "## 🔍 Super-Resolution Comparison"
        )

        st.caption(
            "Comparison of the original image, "
            "conventional Bicubic interpolation, "
            "and the EDSR deep-learning output."
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.subheader(
                "Original"
            )

            st.image(
                lr_rgb,
                caption=(
                    f"Original — "
                    f"{width} × {height}"
                ),
                use_container_width=True
            )

        with col2:

            st.subheader(
                "Bicubic"
            )

            st.image(
                bicubic_rgb,
                caption=(
                    f"Bicubic — "
                    f"{output_width} × "
                    f"{output_height}"
                ),
                use_container_width=True
            )

        with col3:

            st.subheader(
                "EDSR"
            )

            st.image(
                sr_rgb,
                caption=(
                    f"EDSR — "
                    f"{output_width} × "
                    f"{output_height}"
                ),
                use_container_width=True
            )


        # ====================================================
        # FALSE COLOR
        # ====================================================

        st.divider()

        st.markdown(
            "## 🌈 False-Color Comparison"
        )

        st.caption(
            "NIR–Red–Green composite. "
            "Vegetation is emphasized using the B08 NIR band."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Bicubic"
            )

            st.image(
                bicubic_false_color,
                caption="Bicubic NIR–Red–Green",
                use_container_width=True
            )

        with col2:

            st.subheader(
                "EDSR"
            )

            st.image(
                sr_false_color,
                caption="EDSR NIR–Red–Green",
                use_container_width=True
            )


        # ====================================================
        # NDVI
        # ====================================================

        st.divider()

        st.markdown(
            "## 🌱 NDVI Visualization"
        )

        st.caption(
            "NDVI is calculated from B08 NIR and B04 Red. "
            "This is a derived visualization and is not an "
            "additional trained model output."
        )

        ndvi_col1, ndvi_col2 = st.columns(2)

        with ndvi_col1:

            st.subheader(
                "Bicubic NDVI"
            )

            st.image(
                normalize_ndvi_for_display(
                    bicubic_ndvi
                ),
                caption="NDVI — Bicubic",
                use_container_width=True
            )

            st.metric(
                "Mean NDVI",
                f"{np.mean(bicubic_ndvi):.4f}"
            )

        with ndvi_col2:

            st.subheader(
                "EDSR NDVI"
            )

            st.image(
                normalize_ndvi_for_display(
                    sr_ndvi
                ),
                caption="NDVI — EDSR",
                use_container_width=True
            )

            st.metric(
                "Mean NDVI",
                f"{np.mean(sr_ndvi):.4f}"
            )


        # ====================================================
        # SPECTRAL BANDS
        # ====================================================

        st.divider()

        st.markdown(
            "## 🛰️ Spectral Band Visualization"
        )

        st.caption(
            "Individual spectral bands produced by the EDSR model."
        )

        band_columns = st.columns(4)

        for i in range(4):

            with band_columns[i]:

                st.subheader(
                    BAND_NAMES[i]
                )

                st.image(
                    display_band(
                        sr_np[i]
                    ),
                    caption=(
                        f"EDSR "
                        f"{BAND_NAMES[i]}"
                    ),
                    use_container_width=True
                )

                st.caption(
                    f"Min: {sr_np[i].min():.4f}"
                )

                st.caption(
                    f"Max: {sr_np[i].max():.4f}"
                )


        # ====================================================
        # SPECTRAL STATISTICS
        # ====================================================

        st.divider()

        st.markdown(
            "## 📊 Spectral Statistics"
        )

        statistics = []

        for i in range(4):

            band = sr_np[i]

            statistics.append(
                {
                    "Band": BAND_NAMES[i],
                    "Minimum": float(
                        np.min(band)
                    ),
                    "Maximum": float(
                        np.max(band)
                    ),
                    "Mean": float(
                        np.mean(band)
                    ),
                    "Std. Deviation": float(
                        np.std(band)
                    ),
                }
            )

        statistics_df = pd.DataFrame(
            statistics
        )

        st.dataframe(
            statistics_df,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.markdown(
            "## 📈 Model Performance"
        )

        st.caption(
            "These are benchmark results calculated on "
            "the project's 541-image test dataset. "
            "They are NOT metrics for the currently uploaded image."
        )

        try:

            benchmark = (
                load_benchmark_metrics()
            )

            if benchmark is None:

                st.warning(
                    "Benchmark metrics file not found."
                )

            else:

                edsr_psnr = benchmark[
                    "edsr_psnr"
                ]

                bicubic_psnr = benchmark[
                    "bicubic_psnr"
                ]

                edsr_ssim = benchmark[
                    "edsr_ssim"
                ]

                bicubic_ssim = benchmark[
                    "bicubic_ssim"
                ]

                edsr_mse = benchmark[
                    "edsr_mse"
                ]

                bicubic_mse = benchmark[
                    "bicubic_mse"
                ]

                psnr_improvement = (
                    edsr_psnr
                    - bicubic_psnr
                )

                ssim_improvement = (
                    edsr_ssim
                    - bicubic_ssim
                )

                mse_improvement = (
                    bicubic_mse
                    - edsr_mse
                )


                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Average PSNR",
                        f"{edsr_psnr:.2f} dB",
                        f"{psnr_improvement:+.2f} dB vs Bicubic"
                    )

                with c2:

                    st.metric(
                        "Average SSIM",
                        f"{edsr_ssim:.4f}",
                        f"{ssim_improvement:+.4f} vs Bicubic"
                    )

                with c3:

                    st.metric(
                        "Average MSE",
                        f"{edsr_mse:.6f}",
                        f"{mse_improvement:.6f} lower"
                    )


                st.markdown(
                    "### EDSR vs Bicubic"
                )

                comparison_df = pd.DataFrame(
                    {
                        "Metric": [
                            "PSNR (dB)",
                            "SSIM",
                            "MSE"
                        ],
                        "Bicubic": [
                            bicubic_psnr,
                            bicubic_ssim,
                            bicubic_mse
                        ],
                        "EDSR": [
                            edsr_psnr,
                            edsr_ssim,
                            edsr_mse
                        ]
                    }
                )

                st.dataframe(
                    comparison_df,
                    hide_index=True,
                    use_container_width=True
                )


                if (
                    psnr_improvement > 0
                    and ssim_improvement > 0
                    and mse_improvement > 0
                ):

                    st.success(
                        f"""
                        **Benchmark result:** EDSR outperforms
                        Bicubic interpolation on all three reported
                        metrics in this test set.

                        • PSNR improvement:
                        **+{psnr_improvement:.2f} dB**

                        • SSIM improvement:
                        **+{ssim_improvement:.4f}**

                        • MSE reduction:
                        **{mse_improvement:.6f}**
                        """
                    )

                else:

                    st.info(
                        "The benchmark values have been displayed "
                        "without assuming that EDSR improves every metric."
                    )


                st.caption(
                    f"Benchmark sample count: "
                    f"{benchmark['count']} images."
                )

        except Exception as error:

            st.warning(
                "Unable to load benchmark metrics."
            )

            st.code(
                str(error)
            )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.divider()

        st.markdown(
            "## 💾 Download Super-Resolved GeoTIFF"
        )

        st.success(
            "The output contains four super-resolved "
            "bands: B04, B03, B02 and B08."
        )

        st.download_button(
            label=(
                "⬇️ Download 4-Band "
                "Super-Resolved GeoTIFF"
            ),
            data=output_bytes,
            file_name=(
                "sentinel2_super_resolved.tif"
            ),
            mime="image/tiff",
            use_container_width=True
        )


        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        st.divider()

        st.markdown(
            "## 📦 Output Information"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Bands",
            "4"
        )

        c2.metric(
            "Resolution",
            f"{output_width} × {output_height}"
        )

        c3.metric(
            "Scale Factor",
            "4×"
        )

        c4.metric(
            "Format",
            "GeoTIFF"
        )

        st.caption(
            "The output GeoTIFF uses the finer pixel spacing "
            "corresponding to the 4× scale factor and retains "
            "the source CRS/geospatial transform information."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentinel-2 Super-Resolution | "
    "EDSR Deep Learning Model | "
    "SIH26142"
)