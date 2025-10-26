# Crystal Analysis GUI – Detailed Guide

This guide explains the architecture, how to use the app, how to set up the GUI locally or on a remote GPU server, and how to extend it with plugins.

## Overview
- Web app built with FastAPI + Jinja + Bootstrap + Chart.js.
- Modular inference via model plugins; each plugin folder contains `model.py` with `load()` and `infer()`.
- Tabs: Inference (single image), Preprocess, Outputs (dataset/time slider), Live (real-time streaming).
- Results saved under `data/results/` with overlays and JSON stats.

## Architecture
- `app/main.py`: FastAPI app, endpoints for upload, preprocess, inference, dataset ingestion, frame stats, live streaming.
- `app/model_loader.py`: Loads built-in models (`blob`, optional `yolo`) or plugin models from a folder.
- `app/inference_runner.py`: Routes inference through plugin or classical pipeline; draws detections.
- `app/image_loader.py`: Loading/saving images and preprocessing operations.
- `app/postprocess.py`: Computes statistics from detections (counts, areas, aspect ratios).
- `app/templates/`: HTML templates with Bootstrap tabs.
- `app/static/js/app.js`: Client-side logic for tabs, charts, dataset playback, and live updates.
- `models/example_blob/`: Reference plugin.
- `data/`: Uploads and results (overlays, preprocessed, stream).

## Plugin Model Contract
- Structure: a folder containing `model.py`.
- `load() -> Any` (optional): Initialize and return a model object.
- `infer(model, img) -> List[Dict]`: Run inference and return detections with keys `x, y, w, h, angle`.
- Optional: `config.yaml` with model-specific parameters (loader can be extended to read it).
- Example: `models/example_blob/model.py` shows a working plugin using OpenCV’s SimpleBlobDetector.

## API Endpoints
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

## Using the GUI
1. Create and activate Conda environment (Windows/Linux):
   - `conda create -n crystal python=3.12`
   - `conda activate crystal`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
4. Open `http://127.0.0.1:8000/` in your browser.
3. Inference tab:
   - Upload an image and click on it to run inference.
   - Choose a model via dropdown or activate a plugin with a folder path.
   - Overlay and stats appear on the right.
4. Preprocess tab:
   - Select an uploaded image.
   - Choose an operation (CLAHE, equalize, gradient, grayscale), see result image.
5. Outputs tab:
   - Enter a dataset folder path containing images.
   - Use the time slider to scrub through indexed frames; charts and overlays update.
6. Live tab:
   - Click “Start Live” to connect via WebSocket (fallback to polling).
   - Send a frame with a timestamp to update the chart and overlay.

## Real-Time Streaming
- Use the Live tab or POST to `/stream_frame`.
- The server broadcasts updates over WebSockets to connected clients.
- Polling via `/live_stats` remains available for environments without WebSockets.

## Dataset Playback
- Ingest frames via `POST /ingest_dataset`.
- Time slider scrubs frames; `GET /frame_stats` computes or returns cached overlay/stats.
- Caching reduces latency on repeated scrubbing.

## Remote GPU Server Setup
- On remote: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Local tunnel: `ssh -L 8000:127.0.0.1:8000 user@remote-host`
- Browse `http://127.0.0.1:8000/` locally; uploads and datasets reside on remote.
- Add endpoint auth if exposing beyond SSH; consider HTTPS/TLS.

## Environment
- Conda is the recommended environment manager for this project (Windows/Linux/macOS).
- Create environment: `conda create -n crystal python=3.12`
- Activate: `conda activate crystal`
- Install base deps: `pip install -r requirements.txt`
- Optional GPU (PyTorch CUDA 12.x): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12x`
- Optional YOLO: `pip install ultralytics`

## Extending the App
- Add a new model plugin by creating `models/<your_model>/model.py` and implementing `load()`/`infer()`.
- Optionally add a `config.yaml` and enhance `model_loader` to read it.
- Add new outputs (CSV ingestion, comparisons) via new endpoints and JS chart logic.
- For performance, consider background tasks and caching.

## Running Tests
- Install `pytest` in your environment.
- Run tests from the repo root: `pytest -q`.
- Tests cover postprocessing stats, plugin loading and inference, and core endpoints.

## Troubleshooting
- WebSocket fails: browser falls back to polling; check server logs.
- Image not found: verify it appears under `data/uploads`.
- Plugin load error: ensure `model.py` exists and uses valid imports.