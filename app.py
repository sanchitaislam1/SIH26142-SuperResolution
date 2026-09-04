import os
import io

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt

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
# SETTINGS
# ============================================================

CHECKPOINT = os.path.join(
    "checkpoints",
    "edsr_epoch_20.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
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
# RGB CONVERSION
# ============================================================

def create_rgb(image):

    """
    Sentinel-2 channels:

    Channel 0 = B04 (Red)
    Channel 1 = B03 (Green)
    Channel 2 = B02 (Blue)
    Channel 3 = B08 (NIR)

    RGB = B04, B03, B02
    """

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2]
        ],
        axis=-1
    )

    rgb = np.clip(
        rgb,
        0.0,
        1.0
    )

    return rgb


# ============================================================
# TITLE
# ============================================================

st.title("🛰️ Sentinel-2 Super-Resolution")

st.markdown(
    """
    ### AI-Based Satellite Image Enhancement

    Convert **32 × 32 low-resolution Sentinel-2 imagery**
    into **128 × 128 super-resolved imagery** using a trained
    **EDSR deep learning model**.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Model:** EDSR")
    st.write("**Input:** 4-band Sentinel-2")
    st.write("**Input Size:** 32 × 32")
    st.write("**Output Size:** 128 × 128")
    st.write("**Scale Factor:** 4×")
    st.write("**Residual Blocks:** 8")
    st.write("**Features:** 64")

    st.divider()

    st.write("**Bands used:**")
    st.write("B04 — Red")
    st.write("B03 — Green")
    st.write("B02 — Blue")
    st.write("B08 — NIR")

    st.divider()

    if DEVICE.type == "cuda":
        st.success("GPU available")
    else:
        st.info("Running on CPU")


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model = load_model()

    st.success("EDSR model loaded successfully.")

except Exception as e:

    st.error(
        f"Could not load the model: {e}"
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Sentinel-2 Image")

uploaded_file = st.file_uploader(
    "Upload a 4-band LR Sentinel-2 .npy file",
    type=["npy"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD NUMPY FILE
    # --------------------------------------------------------

    try:

        lr_np = np.load(
            uploaded_file
        ).astype(np.float32)

    except Exception as e:

        st.error(
            f"Unable to read the .npy file: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE SHAPE
    # --------------------------------------------------------

    if lr_np.shape != (4, 32, 32):

        st.error(
            f"""
            Invalid image shape: {lr_np.shape}

            Expected:

            (4, 32, 32)
            """
        )

        st.stop()


    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    st.success("Valid Sentinel-2 image detected.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input Size",
            "32 × 32"
        )

    with col2:
        st.metric(
            "Output Size",
            "128 × 128"
        )

    with col3:
        st.metric(
            "Scale",
            "4×"
        )


    # --------------------------------------------------------
    # SUPER RESOLUTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Generate Super-Resolution",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running EDSR super-resolution..."
        ):

            # Convert to tensor

            lr_tensor = torch.from_numpy(
                lr_np
            )

            lr_tensor = (
                lr_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )


            # ------------------------------------------------
            # EDSR
            # ------------------------------------------------

            with torch.no_grad():

                sr = model(
                    lr_tensor
                )

                sr = torch.clamp(
                    sr,
                    0.0,
                    1.0
                )


            # ------------------------------------------------
            # BICUBIC
            # ------------------------------------------------

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


            # ------------------------------------------------
            # CONVERT TO NUMPY
            # ------------------------------------------------

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


            # ------------------------------------------------
            # RGB IMAGES
            # ------------------------------------------------

            lr_rgb = create_rgb(
                lr_np
            )

            bicubic_rgb = create_rgb(
                bicubic_np
            )

            sr_rgb = create_rgb(
                sr_np
            )


            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            os.makedirs(
                "results",
                exist_ok=True
            )

            output_path = os.path.join(
                "results",
                "streamlit_super_resolved.npy"
            )

            np.save(
                output_path,
                sr_np
            )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.success(
            "Super-resolution completed successfully!"
        )

        st.divider()

        st.header(
            "🔍 Super-Resolution Result"
        )


        # ----------------------------------------------------
        # THREE IMAGE COMPARISON
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.subheader(
                "Low Resolution"
            )

            st.image(
                lr_rgb,
                caption="Original LR — 32 × 32",
                use_container_width=True
            )


        with col2:

            st.subheader(
                "Bicubic Baseline"
            )

            st.image(
                bicubic_rgb,
                caption="Bicubic — 128 × 128",
                use_container_width=True
            )


        with col3:

            st.subheader(
                "EDSR Output"
            )

            st.image(
                sr_rgb,
                caption="EDSR — 128 × 128",
                use_container_width=True
            )


        # ====================================================
        # DOWNLOAD RESULT
        # ====================================================

        st.divider()

        st.header(
            "💾 Download Super-Resolved Image"
        )

        buffer = io.BytesIO()

        np.save(
            buffer,
            sr_np
        )

        buffer.seek(0)

        st.download_button(
            label="⬇️ Download 4-Band Super-Resolved .npy",
            data=buffer,
            file_name="sentinel2_super_resolved.npy",
            mime="application/octet-stream",
            use_container_width=True
        )


        # ====================================================
        # OUTPUT INFORMATION
        # ====================================================

        st.divider()

        st.header(
            "📊 Output Information"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Bands",
                "4"
            )

        with col2:
            st.metric(
                "Resolution",
                "128 × 128"
            )

        with col3:
            st.metric(
                "Scale Factor",
                "4×"
            )

        with col4:
            st.metric(
                "Output Format",
                ".npy"
            )


        st.caption(
            "The EDSR output contains B04, B03, B02 and B08 bands."
        )


# ============================================================
# INSTRUCTIONS WHEN NO FILE IS UPLOADED
# ============================================================

else:

    st.info(
        """
        👆 Upload a Sentinel-2 `.npy` file to begin.

        The file must contain:

        **4 channels × 32 × 32 pixels**

        Channels:

        **B04, B03, B02, B08**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sentinel-2 Super-Resolution | EDSR Deep Learning Model"
)