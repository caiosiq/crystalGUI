# Crystal Analysis GUI – Roadmap and Test Plan

This document captures next steps, tests, and deployment guidance to evolve the GUI into a robust, GPU-accelerated, modular system for crystallization image analysis.

## Goals
- Beautiful, modern UI with low-latency interactions.
- Modular backend: plugin-based models, clear APIs, testable components.
- Real-time analysis from streamed frames and historical dataset playback.
- GPU acceleration (CUDA) for heavy models like YOLO-OBB.

## Next Steps (Implementation)
- Models and Plugins
  - Add a scaffold command to generate a new model plugin folder with `model.py` and a config stub.
  - Add optional `config.yaml` support per-plugin; load in `model_loader.set_current_model()`.
  - Implement errors and status endpoint for plugin load failures with readable messages in UI.
- GPU Integration
  - Integrate PyTorch and Ultralytics YOLO (OBB) with CUDA; add model selection UI and upload for weights.
  - Provide ONNX/TensorRT path for optimized inference where possible.
  - Add device selection in UI (`cpu/cuda:X`).
- Real-Time Pipeline
  - Add WebSocket `/ws/live` for pushing stats/overlay updates; keep `/live_stats` as fallback.
  - Implement a bounded queue/ring buffer for incoming frames to avoid overload.
  - Add rate limiter/backpressure and metrics (fps, processing latency).
- Outputs and Analytics
  - CSV ingestion: parse model outputs (detections, stats) and plot histograms, time-series (areas, aspect ratios). 
  - Multi-model comparison: overlay toggles, side-by-side charts, statistics table.
  - Export results: JSON/CSV, and annotated overlays.
- Dataset Playback
  - Improve timestamp parsing (ISO datetime in filename, numeric fallback).
  - Optional `dataset_index.json` generation with metadata (frame time, source, annotations).
  - Cache per-frame stats/overlay to speed up scrubbing.
- Performance & Reliability
  - Use background tasks for expensive work (Starlette BackgroundTasks or worker process).
  - Introduce caching keyed by frame hash and plugin name.
  - Add structured logging and error reporting.
  - Add endpoint-level auth tokens and upload size limits.

## Testing Plan
- Unit Tests
  - `image_loader`: load/save, grayscale, CLAHE, equalize, gradient; shape/type checks and error cases.
  - `postprocess`: stats computed correctly; empty detections; extreme aspect ratios.
  - `inference_runner`: plugin path (calls `infer`), classical pipeline behavior; YOLO path behind feature flag.
  - `model_loader`: plugin load (valid/invalid); fallback to blob; config handling.
- Integration Tests
  - Upload & inference: `POST /upload` then `POST /inference` writes overlay and JSON to `data/results/`.
  - Preprocess: each operation yields a file in `data/results/preprocessed/`.
  - Plugin: `POST /select_model_folder` then `POST /inference` with `models/example_blob` produces detections.
  - Dataset: `POST /ingest_dataset` -> `GET /dataset_frames` -> `GET /frame_stats` updates overlay/stats.
  - Live: `POST /stream_frame` then `GET /live_stats` reflects latest stats quickly.
- UI/UX Checks
  - Tabs render and switch state correctly; error banners appear for failures.
  - Charts update smoothly; dark theme contrast is accessible.
  - Large frame lists remain scrollable/responsive.
- Performance Benchmarks
  - Single-image latency (CPU/GPU); define target bounds per model.
  - Live throughput: sustained `POST /stream_frame` at 5–20 FPS; ensure UI remains smooth.
  - Dataset scrubbing latency; verify caching effectiveness.
- Resilience & Edge Cases
  - Invalid file formats; corrupt images; missing plugin files.
  - Concurrent requests to `/inference` and `/stream_frame` under load.
  - Oversized uploads rejected with clear messages.
  - Windows/Linux path handling; Python 3.11/3.12 and 3.13 behavior.

## Acceptance Criteria
- All endpoints succeed and produce artifacts under `data/results/` with predictable names.
- Plugin models load via folder path and `models/example_blob` functions as reference.
- Live tab shows frequent updates without stutter at configured refresh rate.
- Dataset time slider updates overlay and chart immediately for indexed frames.
- Errors are handled gracefully and surfaced in UI.

## Plugin Contract (Model Folder)
- Required file: `model.py` in the selected folder path.
- Functions:
  - `load() -> Any` (optional): initialize and return a model object (weights, device, etc.).
  - `infer(model, img) -> List[Dict]`: run inference and return detections.
- Detections format: list of dicts with keys `x, y, w, h, angle` (floats). 
- Optional: `config.yaml` for model-specific settings.
- Example: `models/example_blob/model.py` shows a functional minimal plugin.

## API Overview
- `POST /select_model_folder` with `folder_path`: select a plugin model.
- `POST /load_model` with `name`: set built-in `blob` or `yolo` when available.
- `POST /upload` (multipart): upload an image.
- `POST /preprocess` with `image_name`, `operation`: save processed image.
- `POST /inference` with `image_name`: run current model and save overlay/stats.
- `POST /ingest_dataset` with `dataset_path`: index frames; parse timestamps.
- `GET /dataset_frames`: list indexed frames.
- `GET /frame_stats?frame_name=...`: compute overlay/stats for a frame.
- `POST /stream_frame` with `file`, `timestamp`: ingest a live frame and update live state.
- `GET /live_stats`: fetch last live stats/overlay URL.
- Planned: `WS /ws/live`: push updates to clients in real-time.

## Remote GPU Server & Port Forwarding
Yes, the current structure supports running the app and GPU code on a remote server and accessing it via SSH port forwarding.

- On the remote server:
  - `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1`
- From your local machine:
  - `ssh -L 8000:127.0.0.1:8000 user@remote-gpu-host`
  - Open `http://127.0.0.1:8000/` in your local browser.
- Data flow:
  - Upload images and datasets from the browser; they are stored on the remote server.
  - Stream frames by posting to `POST /stream_frame` over the forwarded port.
- Security:
  - Add auth tokens on write endpoints if exposing beyond SSH.
  - Consider HTTPS/TLS if not tunneling via SSH.

## Environment Guidance (GPU)
- Prefer Conda with Python 3.11/3.12 for reliable wheels and CUDA support.
- Suggested setup:
  - `conda create -n crystal python=3.12`
  - `pip install fastapi uvicorn[standard] jinja2 python-multipart`
  - `pip install numpy opencv-python`
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12x`
  - `pip install ultralytics`
- If using venv, target Python 3.11 for better wheel compatibility.

## Milestones
1) Plugin and GPU integration
   - Plugin scaffold, config loading, YOLO-OBB (CUDA), device selection.
2) Real-time and outputs expansion
   - WebSockets for live updates, CSV ingestion, multi-model comparison.
3) Performance and reliability
   - Background processing, caching, metrics, auth, and limits.
4) Test suite and docs
   - Unit/integration tests, UI tests, deployment docs.