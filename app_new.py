# ============================================================
# SENTINEL-2 SUPER-RESOLUTION PLATFORM
# EDSR Deep Learning Model
# ============================================================

import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

from src.model import EDSR


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentinel-2 Super-Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT = ROOT / "checkpoints" / "edsr_epoch_20.pth"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(31, 100, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 10% 80%,
                rgba(0, 190, 160, 0.08),
                transparent 30%
            ),
            #07111f;
        color: #f5f7fa;
    }


    /* ========================================================
       HIDE STREAMLIT DEFAULT ELEMENTS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #081426 0%,
                #0b1729 100%
            );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #e9eef5;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand {
        padding: 10px 4px 25px 4px;
    }

    .brand-icon {
        font-size: 30px;
        margin-bottom: 5px;
    }

    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 1.5px;
    }

    .brand-subtitle {
        color: #7f94ad;
        font-size: 11px;
        letter-spacing: 2px;
        margin-top: 5px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 15px 0 20px 0;
    }

    .eyebrow {
        color: #54a7ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -2px;
        margin: 0;
        color: #ffffff;
    }

    .hero-description {
        color: #91a5bb;
        font-size: 16px;
        line-height: 1.6;
        max-width: 850px;
        margin-top: 15px;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 9px;

        padding: 9px 15px;

        border-radius: 999px;

        background: rgba(18, 185, 125, 0.08);
        border: 1px solid rgba(18, 185, 125, 0.35);

        color: #45e3a8;

        font-size: 12px;
        font-weight: 700;

        letter-spacing: 1px;
        margin: 10px 0 25px 0;
    }

    .status-dot {
        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #35e39c;

        box-shadow:
            0 0 10px rgba(53, 227, 156, 0.9);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(18, 34, 55, 0.95),
                rgba(10, 24, 41, 0.95)
            );

        border: 1px solid rgba(255,255,255,0.08);

        border-radius: 16px;

        padding: 20px;

        min-height: 115px;

        box-shadow:
            0 12px 30px rgba(0,0,0,0.15);
    }

    div[data-testid="stMetricLabel"] {
        color: #7f94ad !important;
        font-size: 12px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 800;
    }


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 35px;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .section-subtitle {
        color: #71869d;
        font-size: 13px;
        margin-bottom: 20px;
    }


    /* ========================================================
       UPLOAD BOX
       ======================================================== */

    div[data-testid="stFileUploader"] {
        background:
            linear-gradient(
                145deg,
                rgba(14, 29, 48, 0.96),
                rgba(8, 20, 35, 0.96)
            );

        border: 1px dashed rgba(85, 157, 255, 0.45);

        border-radius: 18px;

        padding: 12px;

        box-shadow:
            0 15px 35px rgba(0,0,0,0.18);
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        border-radius: 12px;

        min-height: 48px;

        font-weight: 750;

        border: 1px solid rgba(85, 157, 255, 0.4);

        background:
            linear-gradient(
                135deg,
                #1877f2,
                #2454d8
            );

        color: white;

        box-shadow:
            0 8px 25px rgba(24,119,242,0.22);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 12px 30px rgba(24,119,242,0.32);
    }


    /* ========================================================
       IMAGE CARDS
       ======================================================== */

    .image-card {
        background:
            linear-gradient(
                145deg,
                rgba(14, 29, 48, 0.96),
                rgba(8, 20, 35, 0.96)
            );

        border: 1px solid rgba(255,255,255,0.07);

        border-radius: 18px;

        padding: 15px;

        margin-bottom: 10px;
    }

    .image-card-title {
        font-size: 14px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .image-card-description {
        color: #72879d;
        font-size: 11px;
        margin-bottom: 12px;
    }


    /* ========================================================
       INFORMATION BOX
       ======================================================== */

    .info-box {
        background: rgba(13, 28, 46, 0.75);

        border: 1px solid rgba(255,255,255,0.07);

        border-radius: 14px;

        padding: 16px;

        margin: 10px 0;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #53687d;

        font-size: 11px;

        padding: 30px 0 15px 0;

        letter-spacing: 0.5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    # Use the ACTUAL trained checkpoint.
    # Do not load results/super_resolved.pt.
    checkpoint = ROOT / "checkpoints" / "edsr_epoch_20.pth"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Trained checkpoint not found. Expected:\n"
            f"{checkpoint}"
        )

    model = EDSR(
        in_channels=4,
        out_channels=4,
        features=64,
        num_blocks=8,
        scale=4
    )

    # edsr_epoch_20.pth is a direct PyTorch state dictionary.
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    # Support direct state_dict as well as common checkpoint wrappers.
    if isinstance(checkpoint_data, dict):

        if "state_dict" in checkpoint_data:
            state_dict = checkpoint_data["state_dict"]

        elif "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]

        else:
            state_dict = checkpoint_data

    else:
        raise RuntimeError(
            f"Invalid checkpoint type: "
            f"{type(checkpoint_data).__name__}. "
            "Expected a PyTorch state dictionary."
        )

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(
            "The checkpoint contains no model weights."
        )

    # Remove DataParallel's optional "module." prefix.
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }

    # Make sure the checkpoint belongs to this exact EDSR architecture.
    expected_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())

    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {sorted(missing)[:10]}\n"
            f"Unexpected keys: {sorted(unexpected)[:10]}"
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# RGB CONVERSION
# ============================================================

def create_rgb(image):

    """
    Expected channel order:

    Channel 0 = B04 = Red
    Channel 1 = B03 = Green
    Channel 2 = B02 = Blue
    Channel 3 = B08 = NIR

    Display RGB = B04, B03, B02
    """

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2]
        ],
        axis=-1
    )

    rgb = np.nan_to_num(
        rgb,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    return rgb


# ============================================================
# LOAD MODEL
# ============================================================

try:

    MODEL = load_model()

    model_status = True

except Exception as e:

    MODEL = None

    model_status = False

    model_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">🛰️</div>

            <div class="brand-title">
                SENTINEL-2
            </div>

            <div class="brand-subtitle">
                AI IMAGING PLATFORM
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### MODEL")

    st.write("**EDSR**")
    st.caption("Enhanced Deep Super-Resolution")

    st.markdown("### INPUT")

    st.write("4-band Sentinel-2")
    st.caption("B04 · B03 · B02 · B08")

    st.markdown("### ARCHITECTURE")

    st.write("32 × 32 → 128 × 128")
    st.write("4× upscaling")
    st.write("8 residual blocks")
    st.write("64 feature channels")

    st.divider()

    st.markdown("### SYSTEM")

    if DEVICE.type == "cuda":

        st.success(
            "GPU ACCELERATION"
        )

        st.caption(
            f"CUDA device: {torch.cuda.get_device_name(0)}"
        )

    else:

        st.info(
            "CPU MODE"
        )

        st.caption(
            "CUDA is not available"
        )

    st.divider()

    if model_status:

        st.success(
            "MODEL ONLINE"
        )

    else:

        st.error(
            "MODEL OFFLINE"
        )

        st.caption(
            model_error
        )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="eyebrow">
            REMOTE SENSING · DEEP LEARNING
        </div>

        <div class="hero-title">
            Sentinel-2 Super-Resolution
        </div>

        <div class="hero-description">
            Enhance low-resolution multispectral satellite imagery
            using a trained EDSR deep-learning model.
            Transform <b>32 × 32</b> Sentinel-2 patches into
            <b>128 × 128</b> super-resolved imagery.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

if model_status:

    st.markdown(
        """
        <div class="status">

            <div class="status-dot"></div>

            EDSR MODEL ONLINE

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.error(
        "The EDSR model could not be loaded."
    )

    st.code(
        str(CHECKPOINT)
    )

    st.stop()


# ============================================================
# QUICK MODEL METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "INPUT",
        "32 × 32"
    )

with metric2:
    st.metric(
        "OUTPUT",
        "128 × 128"
    )

with metric3:
    st.metric(
        "UPSCALE",
        "4×"
    )

with metric4:
    st.metric(
        "SPECTRAL BANDS",
        "4"
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Process an image</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Upload a prepared Sentinel-2 multispectral patch
        containing B04, B03, B02 and B08.
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload Sentinel-2 patch",
    type=["npy"],
    help="Required shape: 4 × 32 × 32"
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-box">

        <b>Input requirements</b>

        <br><br>

        NumPy file (.npy)

        <br>

        Shape: <b>4 × 32 × 32</b>

        <br><br>

        Band order:

        <br>

        <b>B04</b> Red &nbsp;·&nbsp;
        <b>B03</b> Green &nbsp;·&nbsp;
        <b>B02</b> Blue &nbsp;·&nbsp;
        <b>B08</b> NIR

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FILE PROCESSING
# ============================================================

else:

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    try:

        lr_np = np.load(
            uploaded_file
        ).astype(np.float32)

    except Exception as e:

        st.error(
            f"Unable to read the NumPy file: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE SHAPE
    # --------------------------------------------------------

    if lr_np.shape != (4, 32, 32):

        st.error(
            f"""
            Invalid input shape: `{lr_np.shape}`

            Expected shape:

            `(4, 32, 32)`
            """
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT VALID
    # --------------------------------------------------------

    st.success(
        "Valid 4-band Sentinel-2 patch detected."
    )


    # --------------------------------------------------------
    # FILE DETAILS
    # --------------------------------------------------------

    file_size_mb = uploaded_file.size / (
        1024 * 1024
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "FILE",
            uploaded_file.name
        )

    with col2:

        st.metric(
            "INPUT",
            "4 × 32 × 32"
        )

    with col3:

        st.metric(
            "SIZE",
            f"{file_size_mb:.2f} MB"
        )

    with col4:

        st.metric(
            "DEVICE",
            DEVICE.type.upper()
        )


    # ========================================================
    # PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">Input preview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Natural-colour visualization generated from
            B04 · B03 · B02.
        </div>
        """,
        unsafe_allow_html=True
    )


    lr_rgb = create_rgb(
        lr_np
    )


    preview_col1, preview_col2 = st.columns(
        [1, 2]
    )

    with preview_col1:

        st.markdown(
            """
            <div class="image-card">

                <div class="image-card-title">
                    INPUT PATCH
                </div>

                <div class="image-card-description">
                    B04 · B03 · B02
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            lr_rgb,
            caption="Original Sentinel-2 · 32 × 32",
            use_container_width=True
        )

    with preview_col2:

        st.markdown(
            """
            <div class="info-box">

            <b>Ready for inference</b>

            <br><br>

            The uploaded patch contains all four
            spectral channels required by the EDSR model.

            <br><br>

            <b>B04</b> — Red

            <br>

            <b>B03</b> — Green

            <br>

            <b>B02</b> — Blue

            <br>

            <b>B08</b> — Near Infrared

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # GENERATE BUTTON
    # ========================================================

    st.markdown(
        '<div class="section-title">Super-resolution inference</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀  Generate Super-Resolved Image",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(
            0,
            text="Preparing input..."
        )

        start_time = time.perf_counter()


        # ----------------------------------------------------
        # CONVERT TO TENSOR
        # ----------------------------------------------------

        lr_tensor = torch.from_numpy(
            lr_np
        ).unsqueeze(0).to(DEVICE)

        progress.progress(
            25,
            text="Running EDSR inference..."
        )


        # ----------------------------------------------------
        # EDSR INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            sr = MODEL(
                lr_tensor
            )

            sr = torch.clamp(
                sr,
                0.0,
                1.0
            )


        progress.progress(
            65,
            text="Generating bicubic baseline..."
        )


        # ----------------------------------------------------
        # BICUBIC BASELINE
        # ----------------------------------------------------

        bicubic = F.interpolate(
            lr_tensor,
            size=(128, 128),
            mode="bicubic",
            align_corners=False
        )

        bicubic = torch.clamp(
            bicubic,
            0.0,
            1.0
        )


        progress.progress(
            85,
            text="Preparing visualization..."
        )


        # ----------------------------------------------------
        # NUMPY CONVERSION
        # ----------------------------------------------------

        sr_np = (
            sr
            .squeeze(0)
            .cpu()
            .numpy()
        )

        bicubic_np = (
            bicubic
            .squeeze(0)
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        bicubic_rgb = create_rgb(
            bicubic_np
        )

        sr_rgb = create_rgb(
            sr_np
        )


        # ----------------------------------------------------
        # SAVE OUTPUT
        # ----------------------------------------------------

        output_path = (
            RESULTS_DIR /
            "streamlit_super_resolved.npy"
        )

        np.save(
            output_path,
            sr_np
        )


        elapsed = (
            time.perf_counter()
            - start_time
        )


        progress.progress(
            100,
            text="Inference completed."
        )

        time.sleep(0.3)

        progress.empty()


        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            f"Super-resolution completed successfully in "
            f"{elapsed:.2f} seconds."
        )


        st.markdown(
            '<div class="section-title">Resolution comparison</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-subtitle">
                Compare the original image, conventional bicubic
                interpolation and the EDSR super-resolved result.
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # THREE IMAGE COMPARISON
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                """
                <div class="image-card">

                    <div class="image-card-title">
                        ORIGINAL
                    </div>

                    <div class="image-card-description">
                        Low-resolution input
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                lr_rgb,
                caption="32 × 32",
                use_container_width=True
            )


        with col2:

            st.markdown(
                """
                <div class="image-card">

                    <div class="image-card-title">
                        BICUBIC BASELINE
                    </div>

                    <div class="image-card-description">
                        Conventional 4× interpolation
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                bicubic_rgb,
                caption="128 × 128",
                use_container_width=True
            )


        with col3:

            st.markdown(
                """
                <div class="image-card">

                    <div class="image-card-title">
                        EDSR OUTPUT
                    </div>

                    <div class="image-card-description">
                        AI super-resolved result
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                sr_rgb,
                caption="128 × 128",
                use_container_width=True
            )


        # ====================================================
        # OUTPUT STATISTICS
        # ====================================================

        st.markdown(
            '<div class="section-title">Output information</div>',
            unsafe_allow_html=True
        )


        stat1, stat2, stat3, stat4 = st.columns(4)


        with stat1:

            st.metric(
                "OUTPUT SIZE",
                "4 × 128 × 128"
            )


        with stat2:

            st.metric(
                "SCALE FACTOR",
                "4×"
            )


        with stat3:

            st.metric(
                "BANDS",
                "4"
            )


        with stat4:

            st.metric(
                "INFERENCE",
                f"{elapsed:.2f}s"
            )


        # ====================================================
        # BAND INFORMATION
        # ====================================================

        st.markdown(
            """
            <div class="info-box">

            <b>Super-resolved spectral bands</b>

            <br><br>

            <b>B04</b> Red &nbsp; · &nbsp;
            <b>B03</b> Green &nbsp; · &nbsp;
            <b>B02</b> Blue &nbsp; · &nbsp;
            <b>B08</b> Near Infrared

            <br><br>

            Output tensor shape:
            <b>(4, 128, 128)</b>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.markdown(
            '<div class="section-title">Export result</div>',
            unsafe_allow_html=True
        )

        buffer = io.BytesIO()

        np.save(
            buffer,
            sr_np
        )

        buffer.seek(0)


        st.download_button(
            label="⬇️  Download 4-Band Super-Resolved .npy",
            data=buffer.getvalue(),
            file_name="sentinel2_super_resolved.npy",
            mime="application/octet-stream",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        SENTINEL-2 SUPER-RESOLUTION
        &nbsp; • &nbsp;
        EDSR DEEP LEARNING
        &nbsp; • &nbsp;
        MULTISPECTRAL IMAGE ENHANCEMENT

    </div>
    """,
    unsafe_allow_html=True
)