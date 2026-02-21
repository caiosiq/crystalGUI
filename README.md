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
    - js/app.js
    - js/synth.js: Frontend logic for the Synthesis tab. On DOMContentLoaded, it fetches /synth_default_config, lets you edit parameters via a form, previews single images (/synth_preview), regenerates rows (/synth_preview_bulk), and starts batch generation (/synth_batch). It also supports presets via endpoints listed below.
- data_generator/
  - synth.py: OpenCV/Pillow‑based synthetic image renderer. Implements background formation, the DIC‑style rod shader, ghost rods, debris, optional fused crystals, scale legend, RNG/seeding, t‑parameter scheduling, and oriented bounding box recording. Exposes generate_image(), default_config(), sample_lambda(), lambda_to_t(), and params_for_t().
  - batch_job.py: CLI for batched dataset generation. Produces images plus DOTA quadrilateral labels and YOLO‑OBB labels.
- data/: Project data folders (uploads, results, preprocessed, etc.); created at runtime.
- models/: Place inference model files here; selected via /load_model or /select_model_folder.
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

Synthetic image config, presets, and generation:
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

### Key Capabilities

*   **Physically Accurate Optics**: Simulates DIC (Differential Interference Contrast), Brightfield, Polarization (Birefringence), Fluorescence, and Laser Backscatter (PVM/FBRM).
*   **GPU Acceleration**: Fully vectorized PyTorch pipeline allowing for real-time generation of thousands of particles.
*   **Differentiable**: Supports inverse rendering and parameter optimization.
*   **Complex Morphology**: Generates Rods, Plates, Cubes, Spheres, and procedurally sculpted **Euhedral Polyhedra** (minerals).
*   **Advanced Artifacts**: Simulates sensor noise, blur, chromatic aberration, fouling, bubbles, and droplets.

### Usage in GUI

The **Synthesis Tab** (OSOG Playground) in the GUI provides a user-friendly interface to:
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
