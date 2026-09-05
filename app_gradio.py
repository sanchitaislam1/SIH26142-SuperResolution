import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
import rasterio
from rasterio.io import MemoryFile
from src.model import EDSR

CHECKPOINT = os.path.join("checkpoints", "edsr_epoch_20.pth")
METRICS_FILE = os.path.join("results", "metrics_541.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 4
PATCH_SIZE = 32


def load_model():
    model = EDSR(in_channels=4, out_channels=4, features=64, num_blocks=8, scale=4)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    return model.to(DEVICE).eval()


try:
    model = load_model()
    MODEL_STATUS = f"Model loaded successfully • Device: {DEVICE}"
except Exception as e:
    model = None
    MODEL_STATUS = f"Model loading failed: {e}"


def prepare_bands(data):
    data = data.astype(np.float32)
    if np.nanmax(data) > 1.5:
        data = data / 10000.0
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)


def create_rgb(image):
    rgb = np.stack([image[0], image[1], image[2]], axis=-1)
    low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if high > low:
        rgb = (rgb - low) / (high - low)
    else:
        rgb = np.zeros_like(rgb)
    return np.clip(rgb, 0.0, 1.0)


def display_band(band):
    low, high = np.percentile(band, 2), np.percentile(band, 98)
    if high > low:
        band = (band - low) / (high - low)
    else:
        band = np.zeros_like(band)
    return np.clip(band, 0.0, 1.0)


def run_bicubic_inference(image):
    tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        bicubic = F.interpolate(tensor, scale_factor=SCALE, mode="bicubic", align_corners=False)
    return torch.clamp(bicubic, 0.0, 1.0).squeeze(0).cpu().numpy()


def run_tiled_inference(model, image, progress=gr.Progress()):
    _, height, width = image.shape
    out_h, out_w = height * SCALE, width * SCALE
    output = np.zeros((4, out_h, out_w), dtype=np.float32)
    count = np.zeros((1, out_h, out_w), dtype=np.float32)
    total = int(np.ceil(height / PATCH_SIZE)) * int(np.ceil(width / PATCH_SIZE))
    done = 0

    with torch.inference_mode():
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                patch = image[:, y:min(y + PATCH_SIZE, height), x:min(x + PATCH_SIZE, width)]
                ph, pw = patch.shape[1], patch.shape[2]
                tensor = torch.from_numpy(patch).unsqueeze(0)
                pad_h, pad_w = PATCH_SIZE - ph, PATCH_SIZE - pw
                if pad_h or pad_w:
                    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect" if ph > 1 and pw > 1 else "replicate")
                sr = torch.clamp(model(tensor.to(DEVICE)), 0.0, 1.0).squeeze(0).cpu().numpy()
                valid_h, valid_w = ph * SCALE, pw * SCALE
                oy, ox = y * SCALE, x * SCALE
                output[:, oy:oy + valid_h, ox:ox + valid_w] += sr[:, :valid_h, :valid_w]
                count[:, oy:oy + valid_h, ox:ox + valid_w] += 1.0
                done += 1
                progress(done / total, desc=f"Processing patch {done}/{total}")
                del tensor
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    return np.clip(output / np.maximum(count, 1.0), 0.0, 1.0)


def create_geotiff_bytes(sr, source_profile):
    profile = source_profile.copy()
    profile.update(
        driver="GTiff", height=sr.shape[1], width=sr.shape[2], count=4,
        dtype="float32", compress="deflate", predictor=3,
        transform=profile["transform"] * rasterio.Affine.scale(1 / SCALE, 1 / SCALE)
    )
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            names = ["B04 Red", "B03 Green", "B02 Blue", "B08 NIR"]
            for i, name in enumerate(names, 1):
                dst.write(sr[i - 1], i)
                dst.set_band_description(i, name)
        return memfile.read()


def process_geotiff(file_path, progress=gr.Progress()):
    if file_path is None:
        raise gr.Error("Please upload a 4-band Sentinel-2 GeoTIFF.")
    if model is None:
        raise gr.Error(f"EDSR model could not be loaded. {MODEL_STATUS}")

    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        with MemoryFile(raw_bytes) as memfile:
            with memfile.open() as src:
                if src.count != 4:
                    raise gr.Error(f"This GeoTIFF contains {src.count} bands. Exactly 4 bands are required: B04, B03, B02, B08.")
                lr_raw = src.read()
                source_profile = src.profile.copy()
                input_crs = src.crs

        lr_np = prepare_bands(lr_raw)
        _, height, width = lr_np.shape
        out_h, out_w = height * SCALE, width * SCALE
        lr_rgb = create_rgb(lr_np)

        progress(0.05, desc="Running Bicubic baseline...")
        bicubic_rgb = create_rgb(run_bicubic_inference(lr_np))
        progress(0.10, desc="Running EDSR super-resolution...")
        sr_np = run_tiled_inference(model, lr_np, progress)
        sr_rgb = create_rgb(sr_np)

        progress(0.95, desc="Creating GeoTIFF...")
        output_bytes = create_geotiff_bytes(sr_np, source_profile)
        os.makedirs("results", exist_ok=True)
        output_path = os.path.abspath(os.path.join("results", "gradio_super_resolved.tif"))
        with open(output_path, "wb") as f:
            f.write(output_bytes)

        if os.path.exists(METRICS_FILE):
            try:
                df = pd.read_csv(METRICS_FILE)
                ep, bp = df["edsr_psnr"].mean(), df["bicubic_psnr"].mean()
                es, bs = df["edsr_ssim"].mean(), df["bicubic_ssim"].mean()
                em, bm = df["edsr_mse"].mean(), df["bicubic_mse"].mean()
                metrics = (f"**Average PSNR:** {ep:.2f} dB (+{ep-bp:.2f} dB vs Bicubic)\n\n"
                           f"**Average SSIM:** {es:.4f} (+{es-bs:.4f} vs Bicubic)\n\n"
                           f"**Average MSE:** {em:.6f} ({bm-em:.6f} lower than Bicubic)\n\n"
                           f"Benchmark: 541-image test dataset; not metrics for the uploaded image.")
            except Exception as e:
                metrics = f"Unable to load benchmark metrics: {e}"
        else:
            metrics = "Benchmark metrics file not found."

        info = (f"**Input:** {width} × {height}\n\n**Output:** {out_w} × {out_h}\n\n"
                f"**Scale:** 4×\n\n**Bands:** B04, B03, B02, B08\n\n"
                f"**CRS:** {input_crs if input_crs else 'Not present'}")
        progress(1.0, desc="Completed!")
        return (lr_rgb, bicubic_rgb, sr_rgb, display_band(sr_np[0]), display_band(sr_np[1]),
                display_band(sr_np[2]), display_band(sr_np[3]), metrics, info, output_path)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Unable to process the GeoTIFF: {e}")


with gr.Blocks(title="Sentinel-2 Super-Resolution") as demo:
    gr.Markdown("# 🛰️ Sentinel-2 Super-Resolution")
    gr.Markdown("""### AI-Based Satellite Image Enhancement

Upload a **4-band Sentinel-2 GeoTIFF** containing **B04 — Red, B03 — Green, B02 — Blue and B08 — NIR**.

The trained **EDSR deep-learning model** enhances spatial resolution by **4×** and is compared with conventional **Bicubic interpolation**.
""")
    gr.Markdown(f"**Model:** EDSR • **Residual Blocks:** 8 • **Features:** 64 • **Scale:** 4×")
    gr.Markdown(f"**Status:** {MODEL_STATUS}")

    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="📂 Upload Sentinel-2 GeoTIFF", file_types=[".tif", ".tiff"], type="filepath")
            process_button = gr.Button("🚀 Generate 4× Super-Resolution", variant="primary")
        with gr.Column():
            output_info = gr.Markdown("### 📊 Output Information\n\nNo image processed yet.")

    gr.Markdown("## 🔍 Super-Resolution Comparison")
    with gr.Row():
        original_output = gr.Image(label="Original Sentinel-2", type="numpy")
        bicubic_output = gr.Image(label="Bicubic Baseline", type="numpy")
        edsr_output = gr.Image(label="EDSR 4×", type="numpy")

    gr.Markdown("## 🛰️ Spectral Band Visualization")
    with gr.Row():
        b04_output = gr.Image(label="B04 — Red", type="numpy")
        b03_output = gr.Image(label="B03 — Green", type="numpy")
        b02_output = gr.Image(label="B02 — Blue", type="numpy")
        b08_output = gr.Image(label="B08 — NIR", type="numpy")

    gr.Markdown("## 📈 Model Performance")
    metrics_output = gr.Markdown("Benchmark metrics will appear after processing.")
    download_output = gr.File(label="💾 Download Super-Resolved GeoTIFF")

    process_button.click(
        process_geotiff,
        inputs=input_file,
        outputs=[original_output, bicubic_output, edsr_output, b04_output, b03_output, b02_output,
                 b08_output, metrics_output, output_info, download_output],
        show_progress="full"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
