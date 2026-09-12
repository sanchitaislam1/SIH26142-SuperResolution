import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from PIL import Image, ImageEnhance, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import EDSR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "edsr_epoch_20.pth")
SCALE = 4
PATCH_SIZE = 32

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(f"Model checkpoint missing at: {CHECKPOINT}")
        model = EDSR(in_channels=4, out_channels=4, features=64, num_blocks=8, scale=SCALE)
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model = model.to(DEVICE).eval()
        _model = model
    return _model

def prepare_bands(data):
    data = data.astype(np.float32)
    if np.nanmax(data) > 1.5:
        data = data / 10000.0
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(data, 0.0, 1.0)

def save_rgb_preview(image_chw, save_path, is_sr=False):
    rgb = np.stack([image_chw[0], image_chw[1], image_chw[2]], axis=-1)
    low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
    if high > low:
        rgb = (rgb - low) / (high - low)
    else:
        rgb = np.zeros_like(rgb)
    
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)

    if is_sr:
        # 1. Unsharp mask for crisp boundaries and edge contrast
        img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=175, threshold=2))
        
        # 2. Boost local contrast and fine structure sharpness
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.30)

    img.save(save_path, format="PNG")

def run_upscale(input_path, output_tif_path, input_preview_path, output_preview_path):
    with rasterio.open(input_path) as src:
        band_count = src.count
        lr_raw = src.read()
        source_profile = src.profile.copy()
        input_crs = src.crs
        input_transform = src.transform

    lr_np = prepare_bands(lr_raw)

    if band_count != 4:
        padded_lr = np.zeros((4, lr_np.shape[1], lr_np.shape[2]), dtype=np.float32)
        for i in range(4):
            padded_lr[i] = lr_np[min(i, band_count - 1)]
        lr_np = padded_lr
        source_profile["count"] = 4

    # Save 10m input RGB preview for browser (unaltered)
    save_rgb_preview(lr_np, input_preview_path, is_sr=False)

    _, height, width = lr_np.shape
    out_h, out_w = height * SCALE, width * SCALE

    model = get_model()
    output = np.zeros((4, out_h, out_w), dtype=np.float32)
    count = np.zeros((1, out_h, out_w), dtype=np.float32)

    with torch.inference_mode():
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                patch = lr_np[:, y:min(y + PATCH_SIZE, height), x:min(x + PATCH_SIZE, width)]
                ph, pw = patch.shape[1], patch.shape[2]

                tensor = torch.from_numpy(patch).unsqueeze(0)
                pad_h = PATCH_SIZE - ph
                pad_w = PATCH_SIZE - pw
                if pad_h or pad_w:
                    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")

                tensor = tensor.to(DEVICE)
                sr = model(tensor)
                sr = torch.clamp(sr, 0.0, 1.0).squeeze(0).cpu().numpy()

                valid_h = ph * SCALE
                valid_w = pw * SCALE
                oy, ox = y * SCALE, x * SCALE

                output[:, oy:oy + valid_h, ox:ox + valid_w] += sr[:, :valid_h, :valid_w]
                count[:, oy:oy + valid_h, ox:ox + valid_w] += 1.0
                del tensor, sr

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    sr_final = np.clip(output / np.maximum(count, 1.0), 0.0, 1.0)

    # Save 2.5m super-resolved RGB preview for browser (enhanced sharpness & contrast)
    save_rgb_preview(sr_final, output_preview_path, is_sr=True)

    # Export georeferenced GeoTIFF
    new_transform = rasterio.Affine(
        input_transform.a / SCALE, input_transform.b, input_transform.c,
        input_transform.d, input_transform.e / SCALE, input_transform.f
    )
    output_profile = source_profile.copy()
    output_profile.update(
        driver="GTiff",
        height=out_h,
        width=out_w,
        count=4,
        dtype="float32",
        compress="lzw",
        transform=new_transform,
        crs=input_crs
    )

    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
    with rasterio.open(output_tif_path, "w", **output_profile) as dst:
        dst.write(sr_final.astype(np.float32))
        dst.set_band_description(1, "B04 Red - EDSR 4x")
        dst.set_band_description(2, "B03 Green - EDSR 4x")
        dst.set_band_description(3, "B02 Blue - EDSR 4x")
        dst.set_band_description(4, "B08 NIR - EDSR 4x")

    return {
        "input_res": f"{width} × {height}",
        "output_res": f"{out_w} × {out_h}",
        "crs": str(input_crs) if input_crs else "Preserved Local",
        "device": str(DEVICE)
    }