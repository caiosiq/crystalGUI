from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.templating import Jinja2Templates
from pathlib import Path
import shutil
import json
import uuid
import re
import asyncio
import time
import base64
import cv2
import os
import subprocess
import random
import datetime

# Lazy import heavy modules inside endpoints to avoid failing startup


BASE_DIR = Path(__file__).resolve().parent.parent
# Ensure project root is on sys.path so absolute imports like `crystalGUI.*` work
import sys
PROJECT_ROOT = str(BASE_DIR.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
PREPROC_DIR = DATA_DIR / "preprocessed"
STREAM_DIR = RESULTS_DIR / "stream"
MODELS_DIR = BASE_DIR / "models"
SYNTH_PREVIEW_DIR = DATA_DIR / "generated_synth_previews"
SYNTH_JOBS_DIR = DATA_DIR / "synth_jobs"
SYNTH_PRESETS_DIR = DATA_DIR / "synth_presets"

for p in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, PREPROC_DIR, STREAM_DIR, SYNTH_PREVIEW_DIR, SYNTH_JOBS_DIR, SYNTH_PRESETS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Crystal Analysis GUI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Top-level mounts to avoid nested static conflicts
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads_top")
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/static/preprocessed", StaticFiles(directory=str(PREPROC_DIR)), name="preprocessed")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def list_images():
    return sorted([str(p.name) for p in UPLOADS_DIR.glob("*.*") if p.is_file()])


# In-memory live state for last stream result
LIVE_STATE = {"last": None}
# Connected WebSocket clients for live updates
LIVE_CLIENTS = set()


@app.get("/")
async def index(request: Request):
    images = list_images()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

# Some browsers/extensions try to load the Vite dev client by requesting /@vite/client (or its percent-encoded form).
# Our app does not use Vite, so we provide a harmless stub to prevent noisy 404 logs.
@app.get("/@vite/client")
async def vite_client_stub():
    return Response(content="// Vite HMR disabled; stub response", media_type="application/javascript")

@app.get("/%40vite/client")
async def vite_client_stub_encoded():
    return Response(content="// Vite HMR disabled; stub response (encoded)", media_type="application/javascript")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        return {"ok": False, "error": "No filename provided"}
    
    # Sanitize filename to prevent path traversal and ensure valid characters
    filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
    target = UPLOADS_DIR / filename
    
    # Handle duplicate filenames by adding a counter
    counter = 1
    original_target = target
    while target.exists():
        stem = original_target.stem
        suffix = original_target.suffix
        target = UPLOADS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return RedirectResponse(url="/", status_code=303)


@app.post("/preprocess")
async def preprocess(image_name: str = Form(...), operation: str = Form(...)):
    src_path = UPLOADS_DIR / image_name
    if not src_path.exists():
        return {"ok": False, "error": "Image not found"}
    from . import image_loader
    img = image_loader.load_image(str(src_path))
    processed = image_loader.apply_operation(img, operation)
    out_path = PREPROC_DIR / f"{Path(image_name).stem}_{operation}.png"
    image_loader.save_image(str(out_path), processed)
    return {"ok": True, "processed_path": f"/static/preprocessed/{out_path.name}"}


@app.post("/load_model")
async def load_model(name: str = Form(...)):
    from . import model_loader
    model_loader.set_current_model(name)
    model_info = model_loader.get_model_info()
    return {"ok": True, "model": model_info}


@app.post("/select_model_folder")
async def select_model_folder(folder_path: str = Form(...)):
    from . import model_loader
    # Construct full path to the model folder
    full_path = MODELS_DIR / folder_path
    model_loader.set_current_model(str(full_path))
    return {"ok": True, "model": model_loader.get_model_info()}


@app.post("/inference")
async def run_inference(image_name: str = Form(...)):
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    from . import image_loader, model_loader, inference_runner, postprocess
    timings = {}
    t0 = time.perf_counter()
    img = image_loader.load_image(str(img_path))
    timings["load_image"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = model_loader.get_current_model()
    model_info = model_loader.get_model_info()
    timings["find_model"] = time.perf_counter() - t1
    print(f"[TIMING] Using model: {model_info.get('name')} ({model_info.get('type')})")

    t2 = time.perf_counter()
    dets = inference_runner.run(model, img)
    timings["inference_original"] = time.perf_counter() - t2
    print(f"[TIMING] Inference on original image: {timings['inference_original']:.3f}s, detections={len(dets)}")

    t3 = time.perf_counter()
    stats = postprocess.compute_stats(dets)
    timings["compute_stats"] = time.perf_counter() - t3
    print(f"[TIMING] Compute stats: {timings['compute_stats']:.3f}s")

    t4 = time.perf_counter()
    overlay = inference_runner.draw_detections(img, dets)
    timings["draw_overlay"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    overlay_path = RESULTS_DIR / f"{Path(image_name).stem}_overlay.png"
    image_loader.save_image(str(overlay_path), overlay)
    timings["save_overlay"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    result = {
        "image": image_name,
        "detections": dets,
        "stats": stats,
        "overlay_url": f"/static/results/{overlay_path.name}",
        "model_info": model_info,
    }
    result_path = RESULTS_DIR / f"{Path(image_name).stem}_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)
    timings["save_json"] = time.perf_counter() - t6

    print(
        f"[TIMING] TOTAL inference: "
        f"load_image={timings['load_image']:.3f}s, find_model={timings['find_model']:.3f}s, "
        f"inference={timings['inference_original']:.3f}s, draw={timings['draw_overlay']:.3f}s, "
        f"stats={timings['compute_stats']:.3f}s, save_overlay={timings['save_overlay']:.3f}s, "
        f"save_json={timings['save_json']:.3f}s"
    )

    return {"ok": True, **result, "timings": timings}


@app.post("/inference_compare")
async def inference_compare(image_name: str = Form(...), pipeline: str = Form("{}")):
    """Run inference on both original image and a preprocessed variant defined by a pipeline.
    Returns stats and overlay URLs for comparison.
    """
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    # Parse pipeline parameters
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception as e:
        params = {}
    from . import image_loader, model_loader, inference_runner, postprocess
    # Load original and build processed image
    img_orig = image_loader.load_image(str(img_path))
    img_proc = image_loader.apply_pipeline(img_orig, params)
    # Run inference
    model = model_loader.get_current_model()
    dets_orig = inference_runner.run(model, img_orig)
    dets_proc = inference_runner.run(model, img_proc)
    stats_orig = postprocess.compute_stats(dets_orig)
    stats_proc = postprocess.compute_stats(dets_proc)
    # Overlays
    overlay_orig = inference_runner.draw_detections(img_orig, dets_orig)
    overlay_proc = inference_runner.draw_detections(img_proc, dets_proc)
    stem = Path(image_name).stem
    overlay_orig_path = RESULTS_DIR / f"{stem}_orig_overlay.png"
    overlay_proc_path = RESULTS_DIR / f"{stem}_preproc_overlay.png"
    image_loader.save_image(str(overlay_orig_path), overlay_orig)
    image_loader.save_image(str(overlay_proc_path), overlay_proc)
    model_info = model_loader.get_model_info()
    return {
        "ok": True,
        "image": image_name,
        "model_info": model_info,
        "original": {
            "stats": stats_orig,
            "overlay_url": f"/static/results/{overlay_orig_path.name}",
        },
        "processed": {
            "stats": stats_proc,
            "overlay_url": f"/static/results/{overlay_proc_path.name}",
        }
    }


@app.post("/inference_compare_preproc")
async def inference_compare_preproc(
    image_name: str = Form(...),
    pipeline: str = Form("{}"),
    model_folder: str = Form(...),
):
    """Run inference on both original and a preprocessed variant using a per-call (ephemeral) model.
    Does not save any intermediate images to disk; overlays are returned as base64 data URLs.
    """
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    # Parse pipeline parameters
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    from . import image_loader, model_loader, inference_runner, postprocess
    # Load images
    timings = {}
    t0 = time.perf_counter()
    img_orig = image_loader.load_image(str(img_path))
    timings["load_image"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    img_proc = image_loader.apply_pipeline(img_orig, params)
    timings["apply_pipeline"] = time.perf_counter() - t1
    # Load ephemeral model
    try:
        t2 = time.perf_counter()
        eph_model = model_loader.load_model_ephemeral(str(MODELS_DIR / model_folder))
        timings["load_model"] = time.perf_counter() - t2
        print(f"[TIMING] Ephemeral model loaded: {eph_model.get('name')} in {timings['load_model']:.3f}s")
    except Exception as e:
        return {"ok": False, "error": f"Failed to load model: {e}"}
    # Run inference with ephemeral model
    t3 = time.perf_counter()
    dets_orig = inference_runner.run(eph_model, img_orig)
    timings["inference_original"] = time.perf_counter() - t3
    print(f"[TIMING] Inference (original): {timings['inference_original']:.3f}s, dets={len(dets_orig)}")

    t4 = time.perf_counter()
    dets_proc = inference_runner.run(eph_model, img_proc)
    timings["inference_processed"] = time.perf_counter() - t4
    print(f"[TIMING] Inference (processed): {timings['inference_processed']:.3f}s, dets={len(dets_proc)}")
    stats_orig = postprocess.compute_stats(dets_orig)
    stats_proc = postprocess.compute_stats(dets_proc)
    print("[TIMING] Stats computed for both runs")
    # Draw overlays and encode to base64
    t5 = time.perf_counter()
    overlay_orig = inference_runner.draw_detections(img_orig, dets_orig)
    overlay_proc = inference_runner.draw_detections(img_proc, dets_proc)
    timings["draw_overlays"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    # Encode as JPEG to reduce payload size
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    ok1, buf1 = cv2.imencode('.jpg', overlay_orig, jpeg_params)
    ok2, buf2 = cv2.imencode('.jpg', overlay_proc, jpeg_params)
    timings["encode_jpeg"] = time.perf_counter() - t6
    if not ok1 or not ok2:
        return {"ok": False, "error": "Failed to encode overlays"}
    b64_1 = f"data:image/jpeg;base64,{base64.b64encode(buf1.tobytes()).decode('ascii')}"
    b64_2 = f"data:image/jpeg;base64,{base64.b64encode(buf2.tobytes()).decode('ascii')}"
    model_info = {"type": eph_model.get("type"), "name": eph_model.get("name", "Model")}
    if "path" in eph_model:
        model_info["path"] = eph_model["path"]
    result = {
        "ok": True,
        "image": image_name,
        "model_info": model_info,
        "original": {"stats": stats_orig, "overlay_b64": b64_1},
        "processed": {"stats": stats_proc, "overlay_b64": b64_2},
    }
    print(
        f"[TIMING] TOTAL preprocess inference: "
        f"load_image={timings['load_image']:.3f}s, apply_pipeline={timings['apply_pipeline']:.3f}s, "
        f"load_model={timings['load_model']:.3f}s, inf_orig={timings['inference_original']:.3f}s, "
        f"inf_proc={timings['inference_processed']:.3f}s, draw_overlays={timings['draw_overlays']:.3f}s, "
        f"encode_jpeg={timings['encode_jpeg']:.3f}s"
    )
    return {**result, "timings": timings}


@app.post("/preproc_preview")
async def preproc_preview(image_name: str = Form(...), pipeline: str = Form("{}")):
    """Apply preprocessing pipeline and return processed image as base64 data URL without saving."""
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    from . import image_loader
    t0 = time.perf_counter()
    img = image_loader.load_image(str(img_path))
    proc = image_loader.apply_pipeline(img, params)
    apply_t = time.perf_counter() - t0
    t1 = time.perf_counter()
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    ok, buf = cv2.imencode('.jpg', proc, jpeg_params)
    enc_t = time.perf_counter() - t1
    if not ok:
        return {"ok": False, "error": "Failed to encode processed image"}
    b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    print(f"[TIMING] Preproc preview: apply={apply_t:.3f}s, encode={enc_t:.3f}s")
    return {"ok": True, "overlay_b64": b64}


@app.post("/save_preprocessed")
async def save_preprocessed(image_name: str = Form(...), pipeline: str = Form("{}"), desired_name: str | None = Form(None)):
    """Apply the preprocessing pipeline to the selected image and save it under data/preprocessed.
    If desired_name is not provided, the default is <orig_stem>-preprocessed<orig_ext>.
    """
    src_path = UPLOADS_DIR / image_name
    if not src_path.exists():
        return {"ok": False, "error": "Image not found"}
    # Parse pipeline parameters
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    from . import image_loader
    img = image_loader.load_image(str(src_path))
    processed = image_loader.apply_pipeline(img, params)
    # Build output filename
    orig = Path(image_name)
    ext = orig.suffix if orig.suffix else ".png"
    if desired_name:
        # Sanitize desired name
        safe = re.sub(r'[^\w\-_.]', '_', desired_name)
        # Add extension if missing
        if Path(safe).suffix:
            out_name = safe
        else:
            out_name = f"{safe}{ext}"
    else:
        out_name = f"{orig.stem}-preprocessed{ext}"
    out_path = PREPROC_DIR / out_name
    # Handle duplicates by adding a counter
    counter = 1
    base_stem = Path(out_name).stem
    while out_path.exists():
        out_path = PREPROC_DIR / f"{base_stem}_{counter}{ext}"
        counter += 1
    image_loader.save_image(str(out_path), processed)
    return {"ok": True, "saved_url": f"/static/preprocessed/{out_path.name}", "filename": out_path.name}


# Static mounts are defined above; avoid duplicate mounts here.


def extract_timestamp_from_name(name: str) -> float:
    # Extract first floating or integer number from filename for time
    m = re.search(r"([0-9]+\.?[0-9]*)", name)
    return float(m.group(1)) if m else 0.0


@app.post("/ingest_dataset")
async def ingest_dataset(dataset_path: str = Form(...)):
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": "Invalid dataset path"}
    frames = []
    for p in sorted(d.glob("*.*")):
        if p.is_file():
            frames.append({"name": p.name, "time": extract_timestamp_from_name(p.name), "path": str(p)})
    # Save index for reference
    idx_path = RESULTS_DIR / "dataset_index.json"
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f)
    return {"ok": True, "count": len(frames)}


@app.get("/dataset_frames")
async def dataset_frames():
    idx_path = RESULTS_DIR / "dataset_index.json"
    if not idx_path.exists():
        return {"ok": True, "frames": []}
    with idx_path.open("r", encoding="utf-8") as f:
        frames = json.load(f)
    return {"ok": True, "frames": frames}


@app.get("/frame_stats")
async def frame_stats(frame_name: str):
    # Compute on demand and cache result json
    from . import image_loader, model_loader, inference_runner, postprocess
    idx_path = RESULTS_DIR / "dataset_index.json"
    if not idx_path.exists():
        return {"ok": False, "error": "No dataset ingested"}
    # Try to locate in index
    frames = json.load(idx_path.open("r", encoding="utf-8"))
    match = next((f for f in frames if f["name"] == frame_name), None)
    if not match:
        return {"ok": False, "error": "Frame not found"}
    # Caching: reuse previously computed stats/overlay if available
    cache_path = RESULTS_DIR / f"{Path(frame_name).stem}_frame_results.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return {"ok": True, "stats": cached.get("stats", {}), "overlay_url": cached.get("overlay_url", "")}
    # Compute fresh
    img = image_loader.load_image(match["path"])
    model = model_loader.get_current_model()
    dets = inference_runner.run(model, img)
    stats = postprocess.compute_stats(dets)
    overlay = inference_runner.draw_detections(img, dets)
    overlay_path = RESULTS_DIR / f"{Path(frame_name).stem}_overlay.png"
    image_loader.save_image(str(overlay_path), overlay)
    result = {"frame": frame_name, "stats": stats, "overlay_url": f"/static/results/{overlay_path.name}"}
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)
    return {"ok": True, "stats": stats, "overlay_url": result["overlay_url"]}


@app.post("/stream_frame")
async def stream_frame(file: UploadFile = File(...), timestamp: float = Form(0.0)):
    # Save incoming frame
    ext = Path(file.filename).suffix or ".png"
    target = STREAM_DIR / f"{uuid.uuid4().hex}{ext}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # Process immediately and update LIVE_STATE
    try:
        from . import image_loader, model_loader, inference_runner, postprocess
        img = image_loader.load_image(str(target))
        model = model_loader.get_current_model()
        dets = inference_runner.run(model, img)
        stats = postprocess.compute_stats(dets)
        overlay = inference_runner.draw_detections(img, dets)
        overlay_path = STREAM_DIR / f"{target.stem}_overlay.png"
        image_loader.save_image(str(overlay_path), overlay)
        LIVE_STATE["last"] = {"time": timestamp, "stats": stats, "overlay_url": f"/static/results/stream/{overlay_path.name}"}
        # Push updates to connected WebSocket clients
        payload = json.dumps({"ok": True, "last": LIVE_STATE["last"]})
        stale = []
        for ws in list(LIVE_CLIENTS):
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            LIVE_CLIENTS.discard(ws)
    except Exception as e:
        LIVE_STATE["last"] = {"time": timestamp, "error": str(e)}
    return {"ok": True}


@app.get("/live_stats")
async def live_stats():
    return {"ok": True, "last": LIVE_STATE.get("last")}


@app.get("/system_info")
async def system_info():
    """Get system information including GPU availability and current model."""
    from . import model_loader
    
    # Check GPU availability
    gpu_available = False
    gpu_info = "No GPU detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_info = f"GPU {current_device}: {gpu_name} ({gpu_count} total)"
    except ImportError:
        gpu_info = "PyTorch not available"
    except Exception as e:
        gpu_info = f"GPU check failed: {str(e)}"
    
    # Get current model info
    model_info = model_loader.get_model_info()
    
    return {
        "ok": True,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "model_info": model_info
    }


@app.get("/available_models")
async def available_models():
    """Get list of available models from the models folder."""
    models = []
    # Use configured MODELS_DIR to avoid dependency on working directory
    models_dir = MODELS_DIR
    if models_dir.exists():
        for model_folder in models_dir.iterdir():
            if model_folder.is_dir() and (model_folder / "model.py").exists():
                # Check for display name file
                display_name = model_folder.name.replace('_', ' ').title()
                name_file = model_folder / "name.txt"
                if name_file.exists():
                    try:
                        display_name = name_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        pass  # Use default name if file can't be read

                models.append({
                    "id": model_folder.name,
                    "name": display_name,
                    "type": "model",
                    "folder": model_folder.name
                })

    return {"ok": True, "models": models}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    # Register client
    LIVE_CLIENTS.add(websocket)
    # Immediately send last known state if available
    try:
        if LIVE_STATE.get("last"):
            await websocket.send_text(json.dumps({"ok": True, "last": LIVE_STATE["last"]}))
        # Keep connection alive with heartbeat
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        LIVE_CLIENTS.discard(websocket)


@app.get("/get_image")
async def get_image(name: str):
    """Serve an image from the uploads folder by name.
    This avoids relying on nested static mounts when running behind certain proxies.
    """
    path = UPLOADS_DIR / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))


# =================== Synthetic Generator Endpoints ===================
@app.get("/synth_default_config")
async def synth_default_config():
    """Return default config for the simplified in-GUI synthesizer."""
    # If a saved standard preset exists, return it; otherwise return library default
    try:
        std_path = SYNTH_PRESETS_DIR / "standard.json"
        if std_path.exists():
            with std_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return {"ok": True, "config": cfg, "source": "standard"}
        from crystalGUI.data_generator import default_config
        return {"ok": True, "config": default_config(), "source": "library_default"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/synth_save_standard")
async def synth_save_standard(request: Request):
    """Save provided config as the standard (default) for future sessions."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return {"ok": False, "error": "Missing config"}
    try:
        std_path = SYNTH_PRESETS_DIR / "standard.json"
        with std_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return {"ok": True, "saved": str(std_path)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/synth_save_preset")
async def synth_save_preset(request: Request):
    """Save provided config under a given preset name."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}
    name = str(data.get("name", "")).strip()
    cfg = data.get("config")
    if not name:
        return {"ok": False, "error": "Preset name required"}
    if not isinstance(cfg, dict):
        return {"ok": False, "error": "Missing config"}
    # sanitize name for filesystem
    safe = re.sub(r"[^\w\-_.]", "_", name)
    try:
        path = SYNTH_PRESETS_DIR / f"{safe}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return {"ok": True, "saved": str(path), "name": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/synth_presets")
async def synth_presets():
    """List available presets (including 'standard' if present)."""
    try:
        names = []
        for p in SYNTH_PRESETS_DIR.glob("*.json"):
            names.append(p.stem)
        return {"ok": True, "presets": sorted(names)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/synth_get_preset")
async def synth_get_preset(name: str):
    """Get a preset by name (use 'standard' for the default)."""
    safe = re.sub(r"[^\w\-_.]", "_", str(name))
    path = SYNTH_PRESETS_DIR / f"{safe}.json"
    if not path.exists():
        return {"ok": False, "error": "Preset not found"}
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {"ok": True, "config": cfg, "name": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/synth_preview")
async def synth_preview(request: Request):
    """Generate a single preview as base64 given a JSON body with {t, config}."""
    try:
        data = await request.json()
    except Exception:
        # Fallback to form fields if not JSON
        form = await request.form()
        data = {
            "t": float(form.get("t", 0.0)),
            "config": json.loads(form.get("config", "{}")) if form.get("config") else {},
        }
    t = float(data.get("t", 0.0))
    config = data.get("config", {})
    return_obbs = bool(data.get("return_obbs", False))
    # New: allow client to provide a seed; otherwise choose one and return it
    seed_in = data.get("seed", None)
    try:
        from crystalGUI.data_generator import generate_image
        t0 = time.perf_counter()
        if seed_in is None:
            seed_used = random.SystemRandom().randint(0, 2**31 - 1)
        else:
            seed_used = int(seed_in)
        if return_obbs:
            img, obbs = generate_image(config, t, seed=seed_used, return_obbs=True)
        else:
            img = generate_image(config, t, seed=seed_used)
        gen_time = time.perf_counter() - t0
        # Encode as JPEG
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(data.get("quality", 85))]
        t1 = time.perf_counter()
        ok, buf = cv2.imencode('.jpg', img, jpeg_params)
        if not ok:
            return {"ok": False, "error": "Failed to encode preview"}
        enc_time = time.perf_counter() - t1
        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
        resp = {"ok": True, "image_b64": b64, "width": int(img.shape[1]), "height": int(img.shape[0]), "timings": {"generate_s": gen_time, "encode_s": enc_time, "total_s": gen_time + enc_time}}
        if return_obbs:
            resp["obbs"] = obbs
        # Include the seed used to reproduce at higher resolution
        resp["seed_used"] = int(seed_used)
        # Print a concise timing line for server-side monitoring
        try:
            print(f"[TIMING][synth_preview] t={t:.3f} generate={gen_time:.3f}s, encode={enc_time:.3f}s, total={gen_time+enc_time:.3f}s, size={img.shape[1]}x{img.shape[0]}")
        except Exception:
            pass
        return resp
    except ModuleNotFoundError as e:
        # Defensive import fix if launched outside project root
        import sys
        base = Path(__file__).resolve().parent.parent
        proj_root = str(base.parent)
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        try:
            from crystalGUI.data_generator import generate_image
            # Optional parallel workers: allow query param or env override
            try:
                import os
                pw = data.get("parallel_workers")
                if pw is None:
                    env_pw = os.environ.get("SYNTH_PREVIEW_THREADS")
                    pw = int(env_pw) if env_pw is not None else None
                else:
                    pw = int(pw)
            except Exception:
                pw = None
            t0 = time.perf_counter()
            if return_obbs:
                img, obbs = generate_image(config, t, return_obbs=True, parallel_workers=pw)
            else:
                img = generate_image(config, t, parallel_workers=pw)
            gen_time = time.perf_counter() - t0
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(data.get("quality", 85))]
            t1 = time.perf_counter()
            ok, buf = cv2.imencode('.jpg', img, jpeg_params)
            if not ok:
                return {"ok": False, "error": "Failed to encode preview"}
            enc_time = time.perf_counter() - t1
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
            resp = {"ok": True, "image_b64": b64, "width": int(img.shape[1]), "height": int(img.shape[0]), "timings": {"generate_s": gen_time, "encode_s": enc_time, "total_s": gen_time + enc_time}}
            if return_obbs:
                resp["obbs"] = obbs
            try:
                print(f"[TIMING][synth_preview] t={t:.3f} generate={gen_time:.3f}s, encode={enc_time:.3f}s, total={gen_time+enc_time:.3f}s, size={img.shape[1]}x{img.shape[0]}")
            except Exception:
                pass
            return resp
        except Exception as e2:
            return {"ok": False, "error": str(e2)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/synth_preview_bulk")
async def synth_preview_bulk(request: Request):
    """Generate previews for multiple rows concurrently. Body: {rows:[{id, t}], config, quality}.
    Returns {ok, images: {id: b64, ...}, obbs: {id: [...], ...}}"""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}
    rows = data.get("rows", [])
    config = data.get("config", {})
    quality = int(data.get("quality", 85))
    return_obbs = bool(data.get("return_obbs", False))
    # Ensure import works even if app launched outside project root
    try:
        from crystalGUI.data_generator import generate_image
    except ModuleNotFoundError:
        # Try fixing path at runtime
        import sys
        base = Path(__file__).resolve().parent.parent
        proj_root = str(base.parent)
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        from crystalGUI.data_generator import generate_image
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import statistics

    t_bulk0 = time.perf_counter()

    def gen_one(row):
        rid = str(row.get("id"))
        t = float(row.get("t", 0.0))
        try:
            t0 = time.perf_counter()
            # Optional parallel workers per bulk call: row override, request override, or env
            try:
                pw = row.get("parallel_workers")
                if pw is None:
                    pw = data.get("parallel_workers")
                if pw is None:
                    env_pw = os.environ.get("SYNTH_PREVIEW_THREADS")
                    pw = int(env_pw) if env_pw is not None else None
                else:
                    pw = int(pw)
            except Exception:
                pw = None
            # Per-row seed support
            seed_in = row.get("seed")
            seed_used = int(seed_in) if seed_in is not None else random.SystemRandom().randint(0, 2**31 - 1)
            if return_obbs:
                img, obbs = generate_image(config, t, seed=seed_used, return_obbs=True, parallel_workers=pw)
            else:
                img = generate_image(config, t, seed=seed_used, parallel_workers=pw)
            gen_time = time.perf_counter() - t0
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return rid, None, None, {"generate_s": gen_time, "encode_s": None, "total_s": None}, seed_used
            t1 = time.perf_counter()
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
            enc_time = time.perf_counter() - t1
            if return_obbs:
                return rid, b64, obbs, {"generate_s": gen_time, "encode_s": enc_time, "total_s": gen_time + enc_time}, seed_used
            return rid, b64, None, {"generate_s": gen_time, "encode_s": enc_time, "total_s": gen_time + enc_time}, seed_used
        except Exception:
            return rid, None, None, {"generate_s": None, "encode_s": None, "total_s": None}, None

    images = {}
    obbs_by_id = {}
    timings_by_id = {}
    # Track seeds per row id for deterministic re-rendering
    seeds_by_id = {}
    if not rows:
        return {"ok": True, "images": images, "obbs": obbs_by_id, "timings": timings_by_id, "seeds": seeds_by_id}
    max_workers = max(1, min(len(rows), (os.cpu_count() or 2)))
    # Cap workers to avoid oversubscription on small machines
    max_workers = min(max_workers, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(gen_one, r) for r in rows]
        for fut in as_completed(futures):
            rid, b64, obbs, timing, seed_used = fut.result()
            images[rid] = b64
            if return_obbs:
                obbs_by_id[rid] = obbs
            timings_by_id[rid] = timing
            if seed_used is not None:
                seeds_by_id[rid] = int(seed_used)
    t_bulk = time.perf_counter() - t_bulk0
    # Log aggregate stats for monitoring
    try:
        vals = [v["total_s"] for v in timings_by_id.values() if v and v.get("total_s") is not None]
        if vals:
            print(f"[TIMING][synth_preview_bulk] n={len(rows)} workers={max_workers} total={t_bulk:.3f}s mean={statistics.mean(vals):.3f}s min={min(vals):.3f}s max={max(vals):.3f}s")
        else:
            print(f"[TIMING][synth_preview_bulk] n={len(rows)} workers={max_workers} total={t_bulk:.3f}s (no per-row timings)")
    except Exception:
        pass
    return {"ok": True, "images": images, "obbs": obbs_by_id, "timings": timings_by_id, "total_s": t_bulk, "seeds": seeds_by_id}


@app.post("/synth_batch")
async def synth_batch(request: Request):
    """Submit a batch generation job. JSON body with {config, n_images, out_dir}.
    If Slurm (sbatch) is available, write a .slurm script and submit.
    Otherwise, launch a local background process that runs the batch job module.
    """
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}
    config = data.get("config", {})
    n_images = int(data.get("n_images", 100))
    out_dir = data.get("out_dir") or str(DATA_DIR / "generated_batch")
    # Password gating via .env BATCH_PASSWORD (graceful if python-dotenv is not installed)
    try:
        import dotenv
        # Load the .env file from crystalGUI/ explicitly so Slurm partition and password apply
        dotenv.load_dotenv(dotenv_path=str(BASE_DIR / '.env'))
    except Exception:
        pass
    required_pw = os.environ.get("BATCH_PASSWORD", "")
    user_pw = str(data.get("password", ""))
    if required_pw:
        if not user_pw or user_pw != required_pw:
            return {"ok": False, "error": "Invalid batch password"}
    # Seed base and index offset for parallelization
    # Robust parsing: treat None/"" as 0 to avoid TypeError
    seed_base = int(data.get("seed_base") or 0)
    index_offset = int(data.get("index_offset") or 0)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Optional: number of parallel tasks (Slurm array or local multi-process)
    def _to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default
    n_tasks_req = _to_int(data.get("n_tasks", 0), 0)
    n_tasks_env = _to_int(os.environ.get("SYNTH_BATCH_TASKS", 0), 0)
    n_tasks = n_tasks_req if n_tasks_req > 0 else n_tasks_env
    if n_tasks <= 0:
        n_tasks = 1

    # Slurm resource hints via env (optional)
    cpus_per_task = _to_int(os.environ.get("SYNTH_BATCH_CPUS_PER_TASK", 4), 4)
    time_spec = os.environ.get("SYNTH_BATCH_TIME", "02:00:00")
    mem_spec = os.environ.get("SYNTH_BATCH_MEM", "8G")
    # Allow request body to override env partition/qos
    partition = str(data.get("partition", os.environ.get("SYNTH_BATCH_PARTITION", ""))).strip()
    qos = str(data.get("qos", os.environ.get("SYNTH_BATCH_QOS", ""))).strip()

    # Name job directories using timestamp (YYYY_MM_DD_HH_MM). For Slurm runs we will
    # also include the numeric Slurm job id after submission.
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Staging dir used before we know Slurm job id
    job_dir = SYNTH_JOBS_DIR / f"{ts}_staging"
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    # Create slurm script from template
    slurm_script = job_dir / "synth_job.slurm"
    python_exec = os.environ.get("PYTHON_EXECUTABLE", "python")
    # Use absolute package path and set project root for reliable imports
    project_root = str(BASE_DIR.parent)

    # Build Slurm header
    header_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=crystal_synth",
        f"#SBATCH --output={str(SYNTH_JOBS_DIR)}/{ts}_%A/job.out",
        f"#SBATCH --error={str(SYNTH_JOBS_DIR)}/{ts}_%A/job.err",
        f"#SBATCH --time={time_spec}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem_spec}",
    ]
    if n_tasks > 1:
        header_lines.append(f"#SBATCH --array=0-{n_tasks-1}")
    if partition:
        header_lines.append(f"#SBATCH --partition={partition}")
    if qos:
        header_lines.append(f"#SBATCH --qos={qos}")
    header = "\n".join(header_lines)

    # Slurm body with per-task shard computation
    body = f"""
module purge
module load python || true

cd {project_root}
export PYTHONPATH={project_root}

# Limit library threads to avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_TOTAL={n_images}
SEED_BASE={seed_base}
INDEX_OFFSET_BASE={index_offset}
N_TASKS={n_tasks}
SHARD_SIZE=$(( (N_TOTAL + N_TASKS - 1) / N_TASKS ))
TASK_ID=${{SLURM_ARRAY_TASK_ID:-0}}
START_I=$(( TASK_ID * SHARD_SIZE ))
REMAIN=$(( N_TOTAL - START_I ))
N_THIS=$(( REMAIN < SHARD_SIZE ? REMAIN : SHARD_SIZE ))
OFFSET=$(( INDEX_OFFSET_BASE + START_I ))

echo "Starting batch synth task $TASK_ID of $N_TASKS: images=$N_THIS, offset=$OFFSET"
{python_exec} -m crystalGUI.data_generator.batch_job --n-images "$N_THIS" --out-dir "{str(out_path)}" --config-file "{str(cfg_path)}" --seed-base "$SEED_BASE" --index-offset "$OFFSET"
echo "Task finished"
"""
    slurm_contents = header + "\n\n" + body
    with slurm_script.open("w", encoding="utf-8") as f:
        f.write(slurm_contents)

    sbatch = shutil.which("sbatch")
    if sbatch:
        try:
            res = subprocess.run([sbatch, str(slurm_script)], capture_output=True, text=True)
            if res.returncode == 0:
                # Parse job ID from output (e.g., "Submitted batch job 12345")
                m = re.search(r"Submitted batch job\s+(\d+)", res.stdout)
                slurm_id = m.group(1) if m else None
                # Create final job directory named with timestamp and Slurm job id
                if slurm_id:
                    final_dir = SYNTH_JOBS_DIR / f"{ts}_{slurm_id}"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(cfg_path, final_dir / "config.json")
                        shutil.copy2(slurm_script, final_dir / "synth_job.slurm")
                    except Exception:
                        pass
                return {"ok": True, "mode": "slurm", "job_id": slurm_id, "stdout": res.stdout, "tasks": n_tasks, "out_dir": str(out_path)}
            else:
                # Fall back to local if submission failed
                raise RuntimeError(res.stderr or res.stdout)
        except Exception as e:
            # Fall through to local run
            pass

    # Local fallback: spawn background process(es)
    env = os.environ.copy()
    # Ensure PYTHONPATH includes project root for module resolution
    env["PYTHONPATH"] = project_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    # Limit library threads similarly to Slurm script
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", "1")

    if n_tasks <= 1:
        log_path = final_dir / "local_job.log"
        with log_path.open("w") as lf:
            proc = subprocess.Popen([python_exec, "-m", "crystalGUI.data_generator.batch_job", "--n-images", str(n_images), "--out-dir", str(out_path), "--config-file", str(cfg_path), "--seed-base", str(seed_base), "--index-offset", str(index_offset)], cwd=project_root, stdout=lf, stderr=lf, env=env)
        return {"ok": True, "mode": "local", "job_id": f"local-{job_id}", "pid": proc.pid, "log": str(log_path), "out_dir": str(out_path)}
    else:
        # Local array emulation: split into shards
        shard_size = (n_images + n_tasks - 1) // n_tasks
        pids = []
        logs = []
        for task_id in range(n_tasks):
            start_i = task_id * shard_size
            remain = n_images - start_i
            if remain <= 0:
                break
            n_this = remain if remain < shard_size else shard_size
            offset = index_offset + start_i
            log_path = job_dir / f"local_task_{task_id}.log"
            logs.append(str(log_path))
            lf = log_path.open("w")
            proc = subprocess.Popen([python_exec, "-m", "crystalGUI.data_generator.batch_job", "--n-images", str(n_this), "--out-dir", str(out_path), "--config-file", str(cfg_path), "--seed-base", str(seed_base), "--index-offset", str(offset)], cwd=project_root, stdout=lf, stderr=lf, env=env)
            pids.append(proc.pid)
        return {"ok": True, "mode": "local-array", "job_id": f"local-{job_id}", "pids": pids, "logs": logs, "tasks": len(pids), "out_dir": str(out_path)}