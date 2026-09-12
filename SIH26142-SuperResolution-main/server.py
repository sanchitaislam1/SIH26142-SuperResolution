import os
import time
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from worker import run_upscale, get_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
SAMPLE_FOLDER = os.path.join(BASE_DIR, "data")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

try:
    get_model()
except Exception as e:
    print(f"Warning on model load: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results/<path:filename>")
def serve_result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route("/process", methods=["POST"])
def process():
    try:
        is_sample = request.form.get("is_sample") == "true"
        if is_sample:
            input_path = os.path.join(SAMPLE_FOLDER, "test_real.tif")
            if not os.path.exists(input_path):
                tifs = [f for f in os.listdir(SAMPLE_FOLDER) if f.endswith(".tif")]
                if not tifs:
                    return jsonify({"error": "No pre-cached sample GeoTIFF found in /data"}), 400
                input_path = os.path.join(SAMPLE_FOLDER, tifs[0])
        else:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded."}), 400
            file = request.files["file"]
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

        output_tif = os.path.join(RESULTS_FOLDER, "edsr_4x_output.tif")
        input_preview = os.path.join(RESULTS_FOLDER, "input_preview.png")
        output_preview = os.path.join(RESULTS_FOLDER, "output_preview.png")

        start_time = time.time()
        meta = run_upscale(input_path, output_tif, input_preview, output_preview)
        elapsed_sec = round(time.time() - start_time, 2)
        meta["elapsed_time"] = f"{elapsed_sec}s"

        # Cache buster query string to refresh the image preview
        ts = int(time.time())
        return jsonify({
            "status": "success",
            "metadata": meta,
            "input_preview_url": f"/results/input_preview.png?v={ts}",
            "output_preview_url": f"/results/output_preview.png?v={ts}",
            "download_url": "/download"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download")
def download():
    out_file = os.path.join(RESULTS_FOLDER, "edsr_4x_output.tif")
    return send_file(out_file, as_attachment=True, download_name="Sentinel2_EDSR_2.5m.tif")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)