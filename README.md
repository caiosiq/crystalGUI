CrystalGUI: FastAPI GUI for Crystal Imaging and Synthetic Dataset Generation

Overview

CrystalGUI is a FastAPI-based web application and toolkit for:
- Uploading, inspecting, and preprocessing microscopy images
- Running model inference and visualizing per-image statistics
- Live frame streaming with WebSocket updates
- Generating realistic synthetic phase‑contrast crystal images and labels

The synthetic generator is designed to mimic DIC-like (Differential Interference Contrast) imagery of slender, rod‑like crystals. It supports interactive preview in the GUI and reproducible, batched dataset generation on Slurm clusters.


Quick Start

1) Install dependencies
- Python 3.10+
- pip install -r requirements.txt

2) Run locally
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
  Then open http://localhost:8000/

3) Run on Slurm (interactive)
- ./start_gui_interactive.sh -p <partition> -g <gres> -c <cpus> -m <mem> -t <hh:mm:ss> -w <workers> -P <port> -e </path/to/venv>
  The script will request an interactive allocation, activate your environment, and launch uvicorn. It prints an SSH port‑forwarding command you can run from your laptop to reach the compute node.

4) Run on Slurm (batch)
- sbatch run_gui.slurm
  This starts a gunicorn server with multiple uvicorn workers on port 8000.


Repository Layout

- app/
  - main.py: FastAPI application, CORS setup, static mounts, and routes. Includes endpoints for uploads, preprocessing, inference, live streaming, and synthetic image config/presets.
  - static/
    - css/styles.css
    - js/app.js: Frontend logic for inference, preprocessing, outputs, and live streaming tabs.
- data_generator/
  - synth.py: Thin wrapper re-exporting the OSOG engine (`generate_image`, `SynthConfig`, etc.) for backward-compatible imports.
  - batch_job.py: CLI for batched dataset generation. Produces images plus DOTA quadrilateral labels and YOLO‑OBB labels.
- data/: Project data folders (uploads, results, preprocessed, etc.); created at runtime.
- `models/`: Inference plugins; deploy writes to `models/<name>/` (ignored). Commit curated weights only under `models/public/`. Selected via `/load_model` or `/select_model_folder`.
- start_gui_interactive.sh: Interactive Slurm launcher that sets up the environment and starts uvicorn app.main:app.
- run_gui.slurm: Batch Slurm script that starts gunicorn with uvicorn workers.
- requirements.txt: Python dependencies.
- tests/: Project tests (if present).


Key FastAPI Endpoints (high level)

Uploads, preprocessing, and inference:
- /: Home page (renders index.html) and lists uploaded images
- /outputs_upload_folder: Upload a folder (multipart) preserving relative paths
- /outputs_inspect_dataset: Scan a dataset folder; return counts (readable, zero‑size, unreadable)
- /upload: Upload a single image
- /preprocess: Apply a selected preprocessing operation and save
- /preproc_preview: Return a base64 preview of a preprocessing pipeline
- /save_preprocessed: Save the pipeline result to disk
- /inference, /inference_compare, /inference_compare_preproc: Run model inference, compute stats/overlays, return results
- /available_models, /load_model, /select_model_folder: List and set the active inference model
- /system_info: Report GPU availability, model info
- /ws/live, /stream_frame, /live_stats: WebSocket live feed utilities

Synthetic image config, presets, and generation (OSOG Playground + batch jobs):
- /osog_playground: Dedicated OSOG Lab UI for configuring, previewing, and batch-generating synthetic images
- /synth_default_config: Return the default configuration for synthetic generation (either a saved “standard” preset or library defaults)
- /synth_save_standard: Persist a provided configuration as the standard default
- /synth_save_preset: Save a named configuration
- /synth_presets: List available presets
- /synth_get_preset: Fetch a preset by name
- /synth_preview: Render a single image preview for the current form config
- /synth_preview_bulk: Regenerate preview rows in bulk
- /synth_batch: Trigger batched dataset generation


Synthetic Image Generation: OSOG (Optical Synthetic Object Generator)

CrystalGUI now incorporates **OSOG**, a high-performance, GPU-accelerated synthetic data generation engine designed to simulate optical microscopy images.

> **Note**: The legacy OpenCV/Pillow-based renderer has been replaced by OSOG's differentiable PyTorch engine, offering significantly higher realism and performance.

### What is OSOG?

**OSOG (Optical Synthetic Object Generator)** is a differentiable engine for wave-propagation microscopy. Unlike traditional ray-tracing renderers, OSOG simulates the **wave nature of light** (diffraction, interference, phase shifts) to accurately replicate microscopic effects.

### Architecture: The "Zone" Strategy

OSOG employs a sophisticated **Zone-Based Rendering Strategy** to handle high-density suspensions realistically while maintaining ML-safe labels:

| Zone | Description | Rendering Strategy | ML Labeling |
| :--- | :--- | :--- | :--- |
| **1. Focal Plane** | Crystals in focus | Full Optical Rendering (Sharp edges, internal details) | **Labeled** |
| **2. Shallow Depth** | Slightly out-of-focus | Optical Rendering + Physically-based **Circle of Confusion (CoC)** Blur | **Labeled** |
| **3. Deep Soup** | Far background | **Procedural Anisotropic Noise** (No discrete objects) | **Unlabeled** |

This approach solves the "Blurry Rod Paradox" where out-of-focus particles confuse the model. By replacing the far background with a mathematical texture soup, we ensure the model only learns from valid features.

### Key Capabilities

*   **Physically Accurate Optics**: Simulates DIC (Differential Interference Contrast), Brightfield, Polarization (Birefringence), Fluorescence, and Laser Backscatter (Blaze/FBRM).
*   **Procedural "Soup" Backgrounds**: Anisotropic 2.5D noise simulates dense, flowing suspensions without discrete geometry.
*   **Advanced Artifacts**: Simulates sensor noise, blur, chromatic aberration, **Lens Fouling** (dirt/biofilm), bubbles, and droplets.
*   **GPU Acceleration**: Fully vectorized PyTorch pipeline allowing for real-time generation of thousands of particles.
*   **Differentiable**: Supports inverse rendering and parameter optimization.
*   **Complex Morphology**: Generates Rods, Plates, Cubes, Spheres, and procedurally sculpted **Euhedral Polyhedra** (minerals).

### Usage in GUI

The **OSOG Playground** (`/osog_playground`) provides a user-friendly interface to:
1.  **Configure**: Tweak hundreds of physical and optical parameters.
2.  **Preview**: See real-time results of your configuration.
3.  **Generate**: Launch batch generation jobs (local or Slurm) to create massive annotated datasets for AI training.

For a deep dive into the physics and architecture, see the [OSOG README](osog/README.md).

### Configuration & Programmatic Usage

The generator uses a hierarchical configuration system (`SynthConfig`).

```python
from crystalGUI.osog.config import SynthConfig
from crystalGUI.osog.core.pipeline import Pipeline

# 1. Load Config
config = SynthConfig()
config.canvas.use_gpu = True
config.physics.rods.n_rods_rng_lo_hi = (1000, 1500)

# 2. Initialize Pipeline
pipe = Pipeline(config.to_dict())

# 3. Generate
# Returns a numpy array (H, W, 3)
image, labels = pipe.generate(t=0.5, return_obbs=True)
```

### Batch Generation

The `batch_job.py` module has been updated to use OSOG but maintains the same CLI interface for backward compatibility.

```bash
python -m crystalGUI.data_generator.batch_job \
  --n-images 1000 \
  --out-dir data/synth_dataset \
  --config-file data/synth_config.json
```

Output structure:
- `data/synth_dataset/`
  - `images/*.jpg`
  - `labels_dota/*.txt`
  - `labels_yolo_obb/*.txt`
  - `classes.txt`

## Guide & Architecture

### Architecture
- `app/main.py`: FastAPI app, endpoints for upload, preprocess, inference, dataset ingestion, frame stats, live streaming.
- `app/model_loader.py`: Loads built-in models (`blob`, optional `yolo`) or plugin models from a folder.
- `app/inference_runner.py`: Routes inference through plugin or classical pipeline; draws detections.
- `app/image_loader.py`: Loading/saving images and preprocessing operations.
- `app/postprocess.py`: Computes statistics from detections (counts, areas, aspect ratios).
- `app/templates/`: HTML templates with Bootstrap tabs.
- `app/static/js/app.js`: Client-side logic for tabs, charts, dataset playback, and live updates.
- `models/`: Inference plugins and deployed weights.
  - `models/public/`: **Tracked** curated release models (copy here to commit).
  - `models/<name>/`: Local deploy target from Train YOLO (git-ignored).
  - `models/example_blob/`: Reference plugin template.
- `data/`: Uploads and results (overlays, preprocessed, stream).

### Plugin Model Contract
- Structure: a folder containing `model.py`.
- `load() -> Any` (optional): Initialize and return a model object.
- `infer(model, img) -> List[Dict]`: Run inference and return detections with keys `x, y, w, h, angle`.
- Optional: `config.yaml` with model-specific parameters (loader can be extended to read it).
- Example: `models/example_blob/model.py` shows a working plugin using OpenCV’s SimpleBlobDetector.

### API Endpoints
- `POST /upload` (multipart `file`): Upload a single image.
- `POST /preprocess` (`image_name`, `operation`): Apply preprocessing and save output.
- `POST /load_model` (`name`): Select built-in models (`blob`, `yolo*`).
- `POST /select_model_folder` (`folder_path`): Select a plugin model folder.
- `POST /inference` (`image_name`): Run the current model and save overlay and stats.
- `POST /ingest_dataset` (`dataset_path`): Index frames with timestamps.
- `GET /dataset_frames`: Return ingested frames.
- `GET /frame_stats` (`frame_name`): Compute or return cached stats/overlay for a frame.
- `POST /stream_frame` (multipart `file`, `timestamp`): Ingest a live frame; the server processes and updates live state.
- `GET /live_stats`: Fetch the latest live stats (polling fallback).
- `WS /ws/live`: WebSocket to receive push updates for live stats/overlays.

### Using the GUI
1. Create and activate Conda environment (Windows/Linux):
   - `conda create -n crystal python=3.12`
   - `conda activate crystal`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
4. Open `http://127.0.0.1:8000/` in your browser.
5. Inference tab:
   - Upload an image and click on it to run inference.
   - Choose a model via dropdown or activate a plugin with a folder path.
   - Overlay and stats appear on the right.
6. Preprocess tab:
   - Select an uploaded image.
   - Choose an operation (CLAHE, equalize, gradient, grayscale), see result image.
7. Outputs tab:
   - Enter a dataset folder path containing images.
   - Use the time slider to scrub through indexed frames; charts and overlays update.
8. Live tab:
   - Click “Start Live” to connect via WebSocket (fallback to polling).
   - Send a frame with a timestamp to update the chart and overlay.

### Real-Time Streaming
- Use the Live tab or POST to `/stream_frame`.
- The server broadcasts updates over WebSockets to connected clients.
- Polling via `/live_stats` remains available for environments without WebSockets.

### Dataset Playback
- Ingest frames via `POST /ingest_dataset`.
- Time slider scrubs frames; `GET /frame_stats` computes or returns cached overlay/stats.
- Caching reduces latency on repeated scrubbing.

### Remote GPU Server Setup
- On remote: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Local tunnel: `ssh -L 8000:<node>:8000 user@remote-host`
- Browse `http://127.0.0.1:8000/` locally; uploads and datasets reside on remote.
- Add endpoint auth if exposing beyond SSH; consider HTTPS/TLS.

### Extending the App
- Add a new model plugin by creating `models/<your_model>/model.py` and implementing `load()`/`infer()`.
- Optionally add a `config.yaml` and enhance `model_loader` to read it.
- Add new outputs (CSV ingestion, comparisons) via new endpoints and JS chart logic.
- For performance, consider background tasks and caching.

### Running Tests
- Install `pytest` in your environment.
- Run tests from the repo root: `pytest -q`.
- Tests cover postprocessing stats, plugin loading and inference, and core endpoints.

### Troubleshooting
- WebSocket fails: browser falls back to polling; check server logs.
- Image not found: verify it appears under `data/uploads`.
- Plugin load error: ensure `model.py` exists and uses valid imports.
