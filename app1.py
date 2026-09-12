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
# GeoResolve — SIH 2026
# Deep Learning Based Super Resolution Mapping
# ============================================================

APP_NAME = "GeoResolve"
APP_TAGLINE = "Turn satellite imagery into clearer, analysis-ready maps."

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

# Smaller batch on CPU to avoid memory problems on Render.
INFERENCE_BATCH = 4 if DEVICE.type == "cpu" else 8


# ============================================================
# MODEL
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

    model = model.to(DEVICE)
    model.eval()

    return model


try:

    model = load_model()

    MODEL_STATUS = (
        f"Ready • {DEVICE.type.upper()}"
    )

except Exception as e:

    model = None

    MODEL_STATUS = (
        f"Model loading failed: {e}"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_bands(data):

    data = data.astype(np.float32)

    # Sentinel-2 reflectance scaling
    if np.nanmax(data) > 1.5:
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


def normalize_for_display(image):

    low, high = np.percentile(
        image,
        [2, 98]
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

    # Expected order:
    # B04 Red
    # B03 Green
    # B02 Blue
    # B08 NIR

    rgb = np.stack(
        [
            image[0],
            image[1],
            image[2]
        ],
        axis=-1
    )

    return normalize_for_display(rgb)


def display_band(band):

    return normalize_for_display(
        band
    )


# ============================================================
# BICUBIC BASELINE
# ============================================================

def run_bicubic_rgb(image):

    """
    Bicubic is only needed for visual comparison.

    We therefore interpolate RGB rather than all four
    spectral bands. This avoids unnecessary computation.
    """

    rgb = image[:3]

    tensor = (
        torch.from_numpy(rgb)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():

        result = F.interpolate(
            tensor,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )

    result = (
        torch.clamp(
            result,
            0.0,
            1.0
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return np.transpose(
        result,
        (1, 2, 0)
    )


# ============================================================
# TILED EDSR INFERENCE
# ============================================================

def run_tiled_inference(
    model,
    image,
    progress
):

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

    patches = []

    total = (
        int(np.ceil(height / PATCH_SIZE))
        *
        int(np.ceil(width / PATCH_SIZE))
    )

    processed = 0

    # --------------------------------------------------------
    # Build fixed-size patches
    # --------------------------------------------------------

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

            ph = patch.shape[1]
            pw = patch.shape[2]

            tensor = torch.from_numpy(
                patch
            ).unsqueeze(0)

            pad_h = PATCH_SIZE - ph
            pad_w = PATCH_SIZE - pw

            if pad_h or pad_w:

                mode = (
                    "reflect"
                    if ph > 1 and pw > 1
                    else "replicate"
                )

                tensor = F.pad(
                    tensor,
                    (
                        0,
                        pad_w,
                        0,
                        pad_h
                    ),
                    mode=mode
                )

            patches.append(
                (
                    tensor.squeeze(0),
                    y,
                    x,
                    ph,
                    pw
                )
            )

    # --------------------------------------------------------
    # Batched inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for start in range(
            0,
            len(patches),
            INFERENCE_BATCH
        ):

            batch_items = patches[
                start:
                start + INFERENCE_BATCH
            ]

            batch = torch.stack(
                [
                    item[0]
                    for item in batch_items
                ]
            ).to(DEVICE)

            sr_batch = model(
                batch
            )

            sr_batch = torch.clamp(
                sr_batch,
                0.0,
                1.0
            )

            sr_batch = (
                sr_batch
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Put each SR tile into final image
            # ------------------------------------------------

            for i, item in enumerate(
                batch_items
            ):

                _, y, x, ph, pw = item

                valid_h = ph * SCALE
                valid_w = pw * SCALE

                oy = y * SCALE
                ox = x * SCALE

                sr = sr_batch[i]

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

                processed += 1

            progress(
                processed / total,
                desc=(
                    f"Enhancing satellite image "
                    f"• {processed}/{total} tiles"
                )
            )

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
# GEOTIFF OUTPUT
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

        # 4× finer output grid
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

            names = [
                "B04 Red",
                "B03 Green",
                "B02 Blue",
                "B08 NIR"
            ]

            for i, name in enumerate(
                names,
                1
            ):

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
# BENCHMARK METRICS
# ============================================================

def load_benchmark_metrics():

    if not os.path.exists(
        METRICS_FILE
    ):

        return (
            "Benchmark results are currently unavailable."
        )

    try:

        df = pd.read_csv(
            METRICS_FILE
        )

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

        return f"""
### Model benchmark

| Metric | GeoResolve / EDSR | Bicubic | Improvement |
|---|---:|---:|---:|
| **PSNR** | {ep:.2f} dB | {bp:.2f} dB | +{ep-bp:.2f} dB |
| **SSIM** | {es:.4f} | {bs:.4f} | +{es-bs:.4f} |
| **MSE** | {em:.6f} | {bm:.6f} | {bm-em:.6f} lower |

**Evaluation:** 541-image test dataset.

> These are benchmark results from the test dataset, not measurements calculated from the uploaded image.
"""

    except Exception as e:

        return (
            f"Unable to load benchmark metrics: {e}"
        )


# ============================================================
# MAIN PROCESS
# ============================================================

def process_geotiff(
    file_path,
    progress=gr.Progress()
):

    if file_path is None:

        raise gr.Error(
            "Please upload a Sentinel-2 GeoTIFF first."
        )

    if model is None:

        raise gr.Error(
            f"EDSR model is unavailable.\n\n{MODEL_STATUS}"
        )

    try:

        # ----------------------------------------------------
        # Read GeoTIFF
        # ----------------------------------------------------

        progress(
            0.02,
            desc="Reading satellite imagery..."
        )

        with open(
            file_path,
            "rb"
        ) as f:

            raw_bytes = f.read()

        with MemoryFile(
            raw_bytes
        ) as memfile:

            with memfile.open() as src:

                if src.count != 4:

                    raise gr.Error(
                        "This GeoTIFF contains "
                        f"{src.count} bands.\n\n"
                        "GeoResolve currently expects exactly "
                        "4 bands:\n"
                        "B04 Red • B03 Green • "
                        "B02 Blue • B08 NIR"
                    )

                lr_raw = src.read()

                source_profile = (
                    src.profile.copy()
                )

                input_crs = src.crs

                lr_np = prepare_bands(
                    lr_raw
                )

        _, height, width = lr_np.shape

        out_h = height * SCALE
        out_w = width * SCALE

        # ----------------------------------------------------
        # Prepare original preview
        # ----------------------------------------------------

        progress(
            0.08,
            desc="Preparing original image..."
        )

        original_rgb = create_rgb(
            lr_np
        )

        # ----------------------------------------------------
        # Bicubic preview
        # ----------------------------------------------------

        progress(
            0.12,
            desc="Creating conventional baseline..."
        )

        bicubic_rgb = run_bicubic_rgb(
            lr_np
        )

        # ----------------------------------------------------
        # EDSR
        # ----------------------------------------------------

        progress(
            0.15,
            desc="GeoResolve AI enhancement starting..."
        )

        sr_np = run_tiled_inference(
            model,
            lr_np,
            progress
        )

        sr_rgb = create_rgb(
            sr_np
        )

        # ----------------------------------------------------
        # GeoTIFF
        # ----------------------------------------------------

        progress(
            0.96,
            desc="Preparing GIS-ready GeoTIFF..."
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
                "GeoResolve_super_resolved.tif"
            )
        )

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(
                output_bytes
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = load_benchmark_metrics()

        # ----------------------------------------------------
        # Human-readable information
        # ----------------------------------------------------

        crs_text = (
            str(input_crs)
            if input_crs
            else "Not available"
        )

        info = f"""
### Your result

**Input image**  
`{width:,} × {height:,}` pixels

**Output grid**  
`{out_w:,} × {out_h:,}` pixels

**Enhancement**  
**4× spatial upscaling**

**Spectral bands**  
B04 Red • B03 Green • B02 Blue • B08 NIR

**Coordinate system**  
`{crs_text}`

### What this means

GeoResolve creates a **4× finer output grid** from the
medium-resolution satellite input.

The enhanced image is an AI-estimated reconstruction.
It does **not** create new physically measured satellite
observations.

Your GeoTIFF is kept GIS-ready with its spatial reference
information and adjusted transform.
"""

        progress(
            1.0,
            desc="Your satellite image is ready!"
        )

        return (
            original_rgb,
            bicubic_rgb,
            sr_rgb,

            display_band(
                sr_np[0]
            ),

            display_band(
                sr_np[1]
            ),

            display_band(
                sr_np[2]
            ),

            display_band(
                sr_np[3]
            ),

            metrics,
            info,
            output_path
        )

    except gr.Error:

        raise

    except Exception as e:

        raise gr.Error(
            f"Something went wrong while processing "
            f"the satellite image:\n\n{e}"
        )


# ============================================================
# CUSTOM UI
# ============================================================

CUSTOM_CSS = """

/* =========================================================
   GLOBAL
   ========================================================= */

body {
    background:
        radial-gradient(
            circle at 20% 0%,
            rgba(20, 184, 166, 0.08),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(59, 130, 246, 0.08),
            transparent 35%
        ),
        #f7fafc;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}


/* =========================================================
   HEADER
   ========================================================= */

.hero {
    padding: 38px 34px;
    border-radius: 24px;
    margin-bottom: 24px;

    background:
        linear-gradient(
            135deg,
            #0f172a 0%,
            #123c4a 48%,
            #0f766e 100%
        );

    color: white;

    box-shadow:
        0 20px 50px rgba(
            15,
            23,
            42,
            0.18
        );
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.88;
    max-width: 760px;
    line-height: 1.6;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
}

.badge {
    padding: 7px 12px;
    border-radius: 999px;

    background: rgba(
        255,
        255,
        255,
        0.12
    );

    border: 1px solid rgba(
        255,
        255,
        255,
        0.18
    );

    font-size: 13px;
}


/* =========================================================
   WORKFLOW
   ========================================================= */

.step-card {
    border-radius: 18px;
    padding: 18px;
    background: white;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 8px 25px rgba(
            15,
            23,
            42,
            0.05
        );
}

.step-number {
    width: 32px;
    height: 32px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background: #0f766e;
    color: white;

    font-weight: 700;

    margin-right: 8px;
}


/* =========================================================
   UPLOAD AREA
   ========================================================= */

.upload-box {
    border-radius: 20px !important;
    border: 2px dashed #99f6e4 !important;

    background: #f0fdfa !important;

    min-height: 180px;
}


/* =========================================================
   BUTTON
   ========================================================= */

.primary-btn {
    min-height: 56px !important;

    border-radius: 14px !important;

    font-size: 17px !important;
    font-weight: 700 !important;

    background:
        linear-gradient(
            135deg,
            #0f766e,
            #0d9488
        ) !important;

    border: none !important;

    box-shadow:
        0 10px 25px rgba(
            13,
            148,
            136,
            0.22
        );
}


/* =========================================================
   RESULT CARDS
   ========================================================= */

.result-card {
    border-radius: 18px !important;
    overflow: hidden;

    border: 1px solid #e5e7eb !important;

    background: white !important;
}

.result-card img {
    border-radius: 12px;
}


/* =========================================================
   SECTION HEADERS
   ========================================================= */

.section-title {
    font-size: 25px;
    font-weight: 750;

    margin-top: 24px;
    margin-bottom: 6px;

    color: #0f172a;
}

.section-description {
    color: #64748b;
    margin-bottom: 18px;
}


/* =========================================================
   INFO
   ========================================================= */

.info-box {
    border-radius: 18px !important;
    background: white !important;

    border: 1px solid #e5e7eb !important;

    padding: 10px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    padding: 30px 10px;
    color: #64748b;
    font-size: 13px;
}

.footer strong {
    color: #0f766e;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 700px) {

    .hero {
        padding: 28px 22px;
    }

    .hero-title {
        font-size: 32px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

}

"""


# ============================================================
# BUILD APP
# ============================================================

with gr.Blocks(
    title="GeoResolve — Satellite Super-Resolution",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate"
    )
) as demo:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML(
        f"""
        <div class="hero">

            <div class="hero-title">
                🛰️ {APP_NAME}
            </div>

            <div class="hero-subtitle">
                {APP_TAGLINE}
                Upload a 4-band Sentinel-2 GeoTIFF and
                generate a 4× enhanced spatial grid using
                our trained EDSR deep-learning model.
            </div>

            <div class="badge-row">

                <span class="badge">
                    🧠 EDSR Deep Learning
                </span>

                <span class="badge">
                    🔬 4 Spectral Bands
                </span>

                <span class="badge">
                    🔍 4× Enhancement
                </span>

                <span class="badge">
                    🌍 GIS Ready
                </span>

                <span class="badge">
                    📡 Sentinel-2
                </span>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## How GeoResolve works

        Give us your satellite image. We'll handle the
        technical processing for you.
        """
    )

    with gr.Row():

        gr.HTML(
            """
            <div class="step-card">

                <div>
                    <span class="step-number">1</span>
                    <strong>Upload</strong>
                </div>

                <p>
                    Add your 4-band Sentinel-2
                    GeoTIFF.
                </p>

            </div>
            """
        )

        gr.HTML(
            """
            <div class="step-card">

                <div>
                    <span class="step-number">2</span>
                    <strong>Enhance</strong>
                </div>

                <p>
                    EDSR reconstructs a
                    4× finer output grid.
                </p>

            </div>
            """
        )

        gr.HTML(
            """
            <div class="step-card">

                <div>
                    <span class="step-number">3</span>
                    <strong>Explore</strong>
                </div>

                <p>
                    Compare results and
                    download the GeoTIFF.
                </p>

            </div>
            """
        )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 📂 Start with your satellite image

        **Required bands:** B04 Red, B03 Green,
        B02 Blue and B08 NIR.
        """
    )

    input_file = gr.File(
        label="Drop your Sentinel-2 GeoTIFF here",
        file_types=[
            ".tif",
            ".tiff"
        ],
        type="filepath",
        elem_classes=[
            "upload-box"
        ]
    )

    process_button = gr.Button(
        "🚀 Generate Enhanced Satellite Image",
        variant="primary",
        elem_classes=[
            "primary-btn"
        ]
    )


    # --------------------------------------------------------
    # OUTPUT SUMMARY
    # --------------------------------------------------------

    output_info = gr.Markdown(
        """
        ## 📊 Your analysis

        Upload an image and start processing to see
        the output information here.
        """,
        elem_classes=[
            "info-box"
        ]
    )


    # --------------------------------------------------------
    # VISUAL COMPARISON
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 🔍 See the difference

        The three views help you understand what
        GeoResolve is doing.
        """
    )

    gr.Markdown(
        """
        **Original** → your Sentinel-2 input  
        **Bicubic** → conventional mathematical upscaling  
        **GeoResolve** → EDSR deep-learning reconstruction
        """
    )

    with gr.Row():

        original_output = gr.Image(
            label="Original Sentinel-2",
            type="numpy",
            elem_classes=[
                "result-card"
            ]
        )

        bicubic_output = gr.Image(
            label="Bicubic Baseline",
            type="numpy",
            elem_classes=[
                "result-card"
            ]
        )

        edsr_output = gr.Image(
            label="GeoResolve • EDSR 4×",
            type="numpy",
            elem_classes=[
                "result-card"
            ]
        )


    # --------------------------------------------------------
    # SPECTRAL BANDS
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 🌈 Explore the spectral information

        GeoResolve works with four Sentinel-2 bands rather
        than treating the image as ordinary RGB photography.
        """
    )

    with gr.Row():

        b04_output = gr.Image(
            label="B04 • Red",
            type="numpy"
        )

        b03_output = gr.Image(
            label="B03 • Green",
            type="numpy"
        )

    with gr.Row():

        b02_output = gr.Image(
            label="B02 • Blue",
            type="numpy"
        )

        b08_output = gr.Image(
            label="B08 • Near Infrared",
            type="numpy"
        )


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 📈 Model performance
        """
    )

    metrics_output = gr.Markdown(
        """
        Benchmark results will appear after processing.
        """
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    gr.Markdown(
        """
        ## 💾 Take your result with you

        Your output is saved as a GeoTIFF so it can be
        used in GIS and remote-sensing workflows.
        """
    )

    download_output = gr.File(
        label="Download GeoResolve GeoTIFF"
    )


    # --------------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------------

    with gr.Accordion(
        "⚙️ Technical details",
        open=False
    ):

        gr.Markdown(
            f"""
### GeoResolve configuration

| Component | Configuration |
|---|---|
| Model | EDSR |
| Residual blocks | 8 |
| Feature channels | 64 |
| Input channels | 4 |
| Output channels | 4 |
| Scale | 4× |
| Patch size | {PATCH_SIZE} × {PATCH_SIZE} |
| Device | `{DEVICE}` |
| Input | Sentinel-2 GeoTIFF |
| Output | 4-band GeoTIFF |

### Supported bands

- **B04** — Red
- **B03** — Green
- **B02** — Blue
- **B08** — Near Infrared

### Important scientific note

The enhanced product represents an AI-based reconstruction
on a **4× finer output grid**. It should not be interpreted
as newly measured 2.5 m satellite observations.

The model may estimate details that are not directly present
in the original medium-resolution imagery, so geographic
fidelity and validation remain important.
"""
        )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer">

            <strong>GeoResolve</strong>
            • Stellar Nexus
            • SMART INDIA HACKATHON 2026
            • PS 26142

            <br><br>

            Deep Learning Based Super Resolution Mapping
            from Medium Resolution Satellite Imageries

        </div>
        """
    )


    # ========================================================
    # EVENT
    # ========================================================

    process_button.click(
        fn=process_geotiff,

        inputs=[
            input_file
        ],

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