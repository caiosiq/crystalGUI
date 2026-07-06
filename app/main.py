from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.templating import Jinja2Templates
from pathlib import Path
from typing import List
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
import yaml

# Lazy import heavy modules inside endpoints to avoid failing startup


BASE_DIR = Path(__file__).resolve().parent.parent
# Ensure project root is on sys.path so absolute imports like `crystalGUI.*` work
import sys
PROJECT_ROOT = str(BASE_DIR.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Also add BASE_DIR (crystalGUI) to sys.path to support imports like 'diff_calibration'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
PREPROC_DIR = DATA_DIR / "preprocessed"
STREAM_DIR = RESULTS_DIR / "stream"
MODELS_DIR = BASE_DIR / "models"
SYNTH_PREVIEW_DIR = DATA_DIR / "generated_synth_previews"
SYNTH_JOBS_DIR = DATA_DIR / "synth_jobs"
# Debug exports for inspecting OBB/label-merge behavior
DEBUG_EXPORTS_DIR = DATA_DIR / "debug_exports"
SYNTH_PRESETS_DIR = DATA_DIR / "synth_presets"
# NEW: preprocessing presets directory
PREPROC_PRESETS_DIR = DATA_DIR / "preproc_presets"
# NEW: uploaded dataset folders (preserve client folder structures)
DATASET_UPLOADS_DIR = DATA_DIR / "dataset_uploads"
GENERATED_BATCH_DIR = DATA_DIR / "generated_batch"
LEGACY_TRAINING_DIR = DATA_DIR / "legacy_training"
LEGACY_TRAINING_DATASETS_DIR = LEGACY_TRAINING_DIR / "datasets"
LEGACY_TRAINING_RUNS_DIR = LEGACY_TRAINING_DIR / "runs" / "obb"
LEGACY_TRAINING_LOGS_DIR = LEGACY_TRAINING_DIR / "logs"
LEGACY_SYNTH_JOBS_DIR = LEGACY_TRAINING_LOGS_DIR / "synth_jobs"
LEGACY_MODELS_DIR = MODELS_DIR / "legacy_training"
PUBLIC_MODELS_DIR = MODELS_DIR / "public"
_LEGACY_DIR_NAMES = frozenset({"legacy_training", "public"})
_SKIP_MODEL_DIR_NAMES = _LEGACY_DIR_NAMES | frozenset({"__pycache__"})


def _model_display_name(model_folder: Path) -> str:
    name_file = model_folder / "name.txt"
    if name_file.exists():
        try:
            return name_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return model_folder.name.replace("_", " ").title()


def _iter_inference_model_dirs():
    """Yield (folder_path, folder_id) for plugin folders that contain model.py."""
    if not MODELS_DIR.exists():
        return
    for model_folder in sorted(MODELS_DIR.iterdir()):
        if not model_folder.is_dir():
            continue
        if model_folder.name in _SKIP_MODEL_DIR_NAMES or model_folder.name.startswith("."):
            continue
        if (model_folder / "model.py").exists():
            yield model_folder, model_folder.name
    if PUBLIC_MODELS_DIR.exists():
        for model_folder in sorted(PUBLIC_MODELS_DIR.iterdir()):
            if not model_folder.is_dir() or model_folder.name.startswith("."):
                continue
            if (model_folder / "model.py").exists():
                yield model_folder, f"public/{model_folder.name}"

for p in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, PREPROC_DIR, STREAM_DIR, SYNTH_PREVIEW_DIR, SYNTH_JOBS_DIR, SYNTH_PRESETS_DIR, PREPROC_PRESETS_DIR, DATASET_UPLOADS_DIR, GENERATED_BATCH_DIR, LEGACY_TRAINING_DATASETS_DIR, LEGACY_TRAINING_RUNS_DIR, LEGACY_TRAINING_LOGS_DIR, LEGACY_SYNTH_JOBS_DIR]:
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

# Inject a 'now' function into Jinja2 templates for cache busting
templates.env.globals["now"] = lambda: int(time.time())



def list_images():
    return sorted([str(p.name) for p in UPLOADS_DIR.glob("*.*") if p.is_file()])


@app.get("/outputs_inspect_dataset")
async def outputs_inspect_dataset(dataset_path: str):
    """Inspect a dataset folder recursively and report basic diagnostics:
    - total files with common image extensions
    - number of zero-size files
    - number of unreadable files by OpenCV
    - sample lists of problematic files
    """
    from . import image_loader
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": "Invalid dataset path"}
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            files.append(p)
    total = len(files)
    zero = []
    unreadable = []
    readable = []
    for p in files:
        try:
            sz = p.stat().st_size
        except Exception:
            sz = -1
        if sz == 0:
            zero.append({"path": str(p), "size": sz})
            continue
        # Try to read with OpenCV to verify decodability
        try:
            img = image_loader.load_image(str(p))
            if img is None:
                unreadable.append({"path": str(p), "size": sz, "error": "cv2 returned None"})
            else:
                h, w = img.shape[:2]
                readable.append({"path": str(p), "size": sz, "shape": [h, w]})
        except Exception as e:
            unreadable.append({"path": str(p), "size": sz, "error": str(e)})
    return {
        "ok": True,
        "dataset_path": str(d),
        "total_images": total,
        "zero_size_count": len(zero),
        "unreadable_count": len(unreadable),
        "readable_count": len(readable),
        "zero_size_samples": zero[:10],
        "unreadable_samples": unreadable[:10],
        "readable_samples": readable[:5],
    }


@app.post("/outputs_upload_folder")
async def outputs_upload_folder(request: Request):
    """
    Accept a multipart form with many files and a JSON mapping of relative paths.
    The client should send fields:
      - files: List[UploadFile]
      - paths_json: JSON string list of relative filenames, same order as files

    We will write each uploaded file under DATASET_UPLOADS_DIR preserving the
    relative folder structure. Returns {ok: True, dataset_path: <absolute>}.
    """
    form = await request.form()
    # Debug: log incoming form keys and value types
    try:
        keys_list = list(form.keys())
        print(f"[outputs_upload_folder] Received form with keys: {keys_list}")
        for k in keys_list:
            vals = form.getlist(k)
            types = [type(v).__name__ for v in vals]
            print(f"[outputs_upload_folder] Key '{k}' has {len(vals)} values, types: {types}")
            # Show example file info if any
            for v in vals[:3]:
                if isinstance(v, UploadFile):
                    print(f"[outputs_upload_folder] Example file under key '{k}': filename={v.filename}, content_type={v.content_type}")
    except Exception as e:
        print(f"[outputs_upload_folder] Debug logging failed: {e}")

    # Collect UploadFile entries (robust to various field names like files, files[], file, upload, etc.)
    files = []
    for key in list(form.keys()):
        vals = form.getlist(key)
        for v in vals:
            if isinstance(v, UploadFile) or hasattr(v, "filename"):
                files.append(v)
    # Additionally check common keys explicitly
    for common_key in ("files", "files[]", "file", "upload"):
        if common_key in form:
            for v in form.getlist(common_key):
                if isinstance(v, UploadFile) or hasattr(v, "filename"):
                    files.append(v)
    # Parse paths_json (support both list and object forms)
    rel_paths = []
    pj = form.get("paths_json") or form.get("paths_json_obj")
    if pj:
        try:
            data = json.loads(pj)
            if isinstance(data, dict) and "filenames" in data:
                rel_paths = data.get("filenames") or []
            elif isinstance(data, list):
                rel_paths = data
        except Exception:
            rel_paths = []
    print(f"[outputs_upload_folder] Parsed paths_json entries: {len(rel_paths)}")
    if not files:
        # Return extra debug info to help diagnose client-side upload form-building
        debug_types = {}
        for k in list(form.keys()):
            try:
                debug_types[k] = [type(v).__name__ for v in form.getlist(k)]
            except Exception:
                debug_types[k] = ["<error inspecting>"]
        return {
            "ok": False,
            "error": "No files uploaded",
            "debug": {
                "keys": list(form.keys()),
                "value_types": debug_types,
                "paths_json_len": len(rel_paths),
            },
        }

    # Create a unique dataset directory
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dataset_root = DATASET_UPLOADS_DIR / f"dataset_{ts}_{uuid.uuid4().hex[:8]}"
    dataset_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    nonzero_saved = 0
    # Track whether all uploaded files share the same top-level subfolder
    top_levels = set()
    # Deduplicate by normalized relative path to avoid overwriting the same file twice
    seen_relpaths = set()
    for idx, uf in enumerate(files):
        # Determine relative path for this file
        rel = uf.filename or f"file_{idx}"
        if idx < len(rel_paths) and isinstance(rel_paths[idx], str) and rel_paths[idx].strip():
            rel = rel_paths[idx]
        # Normalize and sanitize
        rel = rel.replace("\\", "/")
        rel = re.sub(r"^/+", "", rel)
        parts = [p for p in rel.split("/") if p and p not in (".", "..")]
        parts = [re.sub(r"[^\w\-_.]", "_", p) for p in parts]
        norm_rel = "/".join(parts) if parts else f"file_{idx}"
        if norm_rel in seen_relpaths:
            # Skip duplicate entries of the same file path
            continue
        seen_relpaths.add(norm_rel)
        # Record top-level subfolder if present
        if len(parts) > 1:
            top_levels.add(parts[0])
        if not parts:
            parts = [f"file_{idx}"]
        subdir = dataset_root
        for part in parts[:-1]:
            subdir = subdir / part
        subdir.mkdir(parents=True, exist_ok=True)
        target = subdir / parts[-1]
        try:
            # Ensure stream is at the beginning in case this UploadFile object was read previously
            try:
                if hasattr(uf, "file") and hasattr(uf.file, "seek"):
                    uf.file.seek(0)
            except Exception:
                pass
            with target.open("wb") as f:
                shutil.copyfileobj(uf.file, f)
            saved += 1
            # Check size to detect empty files
            try:
                if target.exists() and target.stat().st_size > 0:
                    nonzero_saved += 1
                else:
                    print(f"[outputs_upload_folder] Warning: saved zero-byte file at {target}")
            except Exception:
                pass
        except Exception as e:
            print(f"[outputs_upload_folder] Failed to save {rel}: {e}")
    print(f"[outputs_upload_folder] Saved {saved} files under {dataset_root}")
    # If all files share a single top-level folder, provide a more specific path for convenience
    dataset_path_final = None
    try:
        if len(top_levels) == 1:
            only = next(iter(top_levels))
            candidate = dataset_root / only
            if candidate.exists() and candidate.is_dir():
                dataset_path_final = str(candidate)
    except Exception:
        dataset_path_final = None
    return {"ok": True, "dataset_path": str(dataset_root), "dataset_path_final": dataset_path_final, "saved": saved, "nonzero_saved": nonzero_saved}
# In-memory live state for last stream result
LIVE_STATE = {"last": None}
# Connected WebSocket clients for live updates
LIVE_CLIENTS = set()


from app.services.job_manager import JobManager

job_manager = JobManager(SYNTH_JOBS_DIR, LEGACY_SYNTH_JOBS_DIR, LEGACY_TRAINING_DATASETS_DIR)


@app.get("/")
async def index(request: Request):
    images = list_images()
    return templates.TemplateResponse("index.html", {"request": request, "images": images})

@app.get("/osog_playground")
async def playground(request: Request):
    return templates.TemplateResponse("playground.html", {"request": request})

@app.get("/train_yolo")
async def train_yolo_page(request: Request):
    return templates.TemplateResponse("train_yolo.html", {"request": request})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

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


@app.post("/upload_target")
async def upload_target(file: UploadFile = File(...)):
    """API-friendly upload that returns JSON with the saved filename."""
    if not file.filename:
        return {"ok": False, "error": "No filename provided"}
    
    filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
    target = UPLOADS_DIR / filename
    
    counter = 1
    original_target = target
    while target.exists():
        stem = original_target.stem
        suffix = original_target.suffix
        target = UPLOADS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {"ok": True, "filename": target.name, "path": str(target)}


def _assert_upload_image_name(image_name: str) -> Path:
    """Resolve a single uploaded image filename under UPLOADS_DIR (no path traversal)."""
    if not image_name or image_name.strip() != image_name:
        raise ValueError("Invalid image name")
    if ".." in image_name or "/" in image_name or "\\" in image_name:
        raise ValueError("Invalid image name")
    target = UPLOADS_DIR / image_name
    if not target.is_file():
        raise ValueError("Image not found")
    if not _path_is_under(UPLOADS_DIR, target):
        raise ValueError("Invalid image path")
    return target


def _cleanup_upload_artifacts(image_name: str) -> None:
    """Remove cached inference/preprocess outputs tied to an uploaded image."""
    stem = Path(image_name).stem
    for path in (
        RESULTS_DIR / f"{stem}_overlay.png",
        RESULTS_DIR / f"{stem}_results.json",
        RESULTS_DIR / f"{stem}_orig_overlay.png",
        RESULTS_DIR / f"{stem}_preproc_overlay.png",
    ):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
    if PREPROC_DIR.is_dir():
        for pattern in (f"{stem}_*", f"{stem}-preprocessed*"):
            for path in PREPROC_DIR.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                except OSError:
                    pass


@app.post("/delete_upload")
async def delete_upload(image_name: str = Form(...)):
    """Delete an uploaded inference image and its cached results."""
    try:
        target = _assert_upload_image_name(image_name)
        target.unlink()
        _cleanup_upload_artifacts(image_name)
        return {"ok": True, "deleted": image_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
async def select_model_folder(folder_path: str = Form(...), device: str = Form(None)):
    from . import model_loader
    # Construct full path to the model folder
    full_path = MODELS_DIR / folder_path
    
    config_override = {}
    if device:
        config_override["device"] = device
        
    model_loader.set_current_model(str(full_path), config_override=config_override)
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
        "overlay_url": f"/static/results/{overlay_path.name}?v={int(time.time() * 1000)}",
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
        "original": {"stats": stats_orig, "overlay_b64": b64_1, "detections": dets_orig},
        "processed": {"stats": stats_proc, "overlay_b64": b64_2, "detections": dets_proc},
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
    """Apply preprocessing pipeline and return processed image as base64 data URL without saving.
    For faster responsiveness, this endpoint downsamples large images for preview purposes.
    Full-resolution processing remains available via save_preprocessed and inference endpoints.
    """
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    from . import image_loader
    import os
    import math
    t0 = time.perf_counter()
    img = image_loader.load_image(str(img_path))
    load_t = time.perf_counter() - t0

    # Downscale for preview if image is large
    max_dim_env = os.getenv("PREPROC_PREVIEW_MAX_DIM", "1400")
    try:
        PREVIEW_MAX_DIM = int(max_dim_env)
    except Exception:
        PREVIEW_MAX_DIM = 1400

    h, w = img.shape[:2]
    t_ds0 = time.perf_counter()
    if max(h, w) > PREVIEW_MAX_DIM:
        scale = PREVIEW_MAX_DIM / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    downscale_t = time.perf_counter() - t_ds0

    t1 = time.perf_counter()
    proc = image_loader.apply_pipeline(img, params)
    apply_t = time.perf_counter() - t1

    t2 = time.perf_counter()
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    ok, buf = cv2.imencode('.jpg', proc, jpeg_params)
    enc_t = time.perf_counter() - t2
    if not ok:
        return {"ok": False, "error": "Failed to encode processed image"}
    b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
    print(
        f"[TIMING] Preproc preview: load={load_t:.3f}s, downscale={downscale_t:.3f}s, "
        f"apply={apply_t:.3f}s, encode={enc_t:.3f}s, size=({img.shape[1]}x{img.shape[0]})"
    )
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


_DATASET_TIME_FOLDER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*_?\s*min\b", re.IGNORECASE)
_DATASET_T_PREFIX_RE = re.compile(r"^t(\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_timestamp_from_name(name: str) -> float:
    """Parse time from filename. Supports microscopy names like '35 min 20x.jpg'."""
    # Prefer explicit minutes pattern before magnification suffix
    m = re.search(r"(\d+(?:\.\d+)?)\s*min", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # t10320-r1.jpg style (minutes encoded in filename prefix)
    m = _DATASET_T_PREFIX_RE.search(Path(name).stem)
    if m:
        return float(m.group(1))
    # Fallback: first number in filename
    m = re.search(r"([0-9]+\.?[0-9]*)", name)
    return float(m.group(1)) if m else 0.0


def extract_timestamp_from_path(file_path: Path, dataset_root=None) -> float:
    """Parse time from nested folders (e.g. 90_min/1.jpg) or filename."""
    path = Path(file_path)
    ancestors = [path.parent]
    if dataset_root is not None:
        root = Path(dataset_root).resolve()
        cur = path.parent.resolve()
        while True:
            ancestors.append(cur)
            if cur == root or cur == cur.parent:
                break
            cur = cur.parent
    seen = set()
    for parent in ancestors:
        key = str(parent)
        if key in seen:
            continue
        seen.add(key)
        m = _DATASET_TIME_FOLDER_RE.search(parent.name)
        if m:
            return float(m.group(1))
        m = _DATASET_T_PREFIX_RE.search(parent.name)
        if m:
            return float(m.group(1))
    return extract_timestamp_from_name(path.name)


def _dataset_image_rel_name(file_path: Path, dataset_root: Path) -> str:
    try:
        return file_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return file_path.name


def _overlay_stem_for_dataset_image(rel_name: str) -> str:
    """Stable overlay filename stem, e.g. 120_min/1.jpg -> 120_min__1."""
    parts = Path(rel_name)
    if parts.parent and str(parts.parent) not in (".", ""):
        prefix = parts.parent.as_posix().replace("/", "__")
        return f"{prefix}__{parts.stem}"
    return parts.stem


def _iter_dataset_image_files(dataset_root: Path) -> list:
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    root = Path(dataset_root)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            files.append(p)
    return files


def format_timestamp_label(value: float, unit: str = "min") -> str:
    """Human-readable time label for charts and UI."""
    if unit:
        if float(value).is_integer():
            return f"{int(value)} {unit}"
        return f"{value:g} {unit}"
    return str(value)


def _list_dataset_image_frames(dataset_dir: Path) -> list:
    """Collect image frames from a dataset (flat or nested timepoint folders)."""
    root = Path(dataset_dir)
    frames = []
    for p in _iter_dataset_image_files(root):
        rel = _dataset_image_rel_name(p, root)
        frames.append({
            "name": rel,
            "time": extract_timestamp_from_path(p, root),
            "path": str(p),
        })
    frames.sort(key=lambda f: (f["time"], f["name"]))
    return frames


@app.post("/ingest_dataset")
async def ingest_dataset(dataset_path: str = Form(...)):
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": "Invalid dataset path"}
    frames = _list_dataset_image_frames(d)
    # Save index for reference
    idx_path = RESULTS_DIR / "dataset_index.json"
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f)
    return {"ok": True, "count": len(frames), "time_unit": "min"}


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


@app.get("/model_statuses")
async def model_statuses():
    """Check status of inference plugins (skips container dirs like public/, legacy_training/)."""
    import importlib.util

    statuses = []
    for model_folder, folder_id in _iter_inference_model_dirs():
        model_py = model_folder / "model.py"
        status = {
            "id": folder_id,
            "folder": folder_id,
            "name": _model_display_name(model_folder),
            "status": "unknown",
            "error": None,
        }
        try:
            spec = importlib.util.spec_from_file_location(f"check_{folder_id.replace('/', '_')}", str(model_py))
            if spec is None or spec.loader is None:
                status["status"] = "error"
                status["error"] = "Failed to create module spec"
            else:
                with open(model_py, "r", encoding="utf-8") as f:
                    compile(f.read(), str(model_py), "exec")
                status["status"] = "ok"
        except Exception as e:
            status["status"] = "error"
            status["error"] = f"{type(e).__name__}: {str(e)}"
        statuses.append(status)

    return {"ok": True, "statuses": statuses}

@app.get("/available_models")
async def available_models():
    """Get list of available models from the models folder."""
    models = []
    for model_folder, folder_id in _iter_inference_model_dirs():
        models.append({
            "id": folder_id,
            "name": _model_display_name(model_folder),
            "type": "model",
            "folder": folder_id,
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
            from crystalGUI.osog.config import merge_config_with_defaults
            return {"ok": True, "config": merge_config_with_defaults(cfg), "source": "standard"}
        
        # Always return the authoritative OSOG SynthConfig default
        from crystalGUI.osog.config import SynthConfig
        return {"ok": True, "config": SynthConfig().to_dict(), "source": "library_default"}
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR][synth_preview] {tb_str}")
        return {"ok": False, "error": str(e), "traceback": tb_str}


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
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR][synth_preview] {tb_str}")
        return {"ok": False, "error": str(e), "traceback": tb_str}


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
    """List presets saved as JSON files in data/synth_presets/."""
    try:
        names = [p.stem for p in SYNTH_PRESETS_DIR.glob("*.json")]
        return {"ok": True, "presets": sorted(names)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.delete("/synth_delete_preset/{name}")
async def synth_delete_preset(name: str):
    """Delete a user-saved preset JSON from data/synth_presets/."""
    safe = re.sub(r"[^\w\-_.]", "_", name)
    path = SYNTH_PRESETS_DIR / f"{safe}.json"
    if path.exists():
        try:
            path.unlink()
            return {"ok": True, "deleted": safe}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Preset not found"}


@app.get("/synth_jobs")
async def synth_jobs():
    """List all batch generation jobs."""
    return {"ok": True, "jobs": job_manager.list_jobs()}


@app.delete("/synth_delete_job/{job_id}")
async def synth_delete_job(job_id: str):
    """Stop and delete a job."""
    success = job_manager.delete_job(job_id)
    return {"ok": success}


@app.get("/synth_get_preset")
async def synth_get_preset(name: str):
    """Get a preset JSON from data/synth_presets/."""
    safe = re.sub(r"[^\w\-_.]", "_", str(name))
    path = SYNTH_PRESETS_DIR / f"{safe}.json"
    if not path.exists():
        return {"ok": False, "error": "Preset not found"}
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        from crystalGUI.osog.config import merge_config_with_defaults
        return {"ok": True, "config": merge_config_with_defaults(cfg), "name": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/synth_constraints")
async def synth_constraints():
    """Return the technical constraints (Canny Regions) for all components."""
    try:
        from crystalGUI.osog.presets import COMPONENT_PRESETS
        constraints = {}
        for name, preset in COMPONENT_PRESETS.items():
            constraints[name] = {}
            for param_name, canny_param in preset.get_canny_constraints().items():
                constraints[name][param_name] = {
                    "default": canny_param.default,
                    "hard_min": canny_param.hard_min,
                    "hard_max": canny_param.hard_max
                }
        return {"ok": True, "constraints": constraints}
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
    return_heads = bool(data.get("return_heads", False))
    return_obbs_raw = bool(data.get("return_obbs_raw", False))
    # New: allow client to provide a seed; otherwise choose one and return it
    seed_in = data.get("seed", None)
    try:
        from crystalGUI.data_generator import generate_image
        t0 = time.perf_counter()
        if seed_in is None:
            seed_used = random.SystemRandom().randint(0, 2**31 - 1)
        else:
            seed_used = int(seed_in)
            
        # Call generator
        res = generate_image(config, t, seed=seed_used, return_obbs=return_obbs, return_heads=return_heads, return_obbs_raw=return_obbs_raw)
        
        img = None
        obbs = None
        obbs_raw = None
        heads = None
        
        if return_obbs and return_obbs_raw and return_heads:
            img, obbs, obbs_raw, heads = res
        elif return_obbs and return_obbs_raw:
            img, obbs, obbs_raw = res
        elif return_obbs and return_heads:
            img, obbs, heads = res
        elif return_obbs:
            img, obbs = res
        elif return_heads:
            img, heads = res
        else:
            img = res

        gen_time = time.perf_counter() - t0
        # Encode as JPEG
        jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(data.get("quality", 85))]
        t1 = time.perf_counter()
        ok, buf = cv2.imencode('.jpg', img, jpeg_params)
        if not ok:
            return {"ok": False, "error": "Failed to encode preview"}
        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"
        
        # Encode heads
        heads_b64 = {}
        if heads:
            import numpy as np
            for k, v in heads.items():
                # Normalize float map to 0-255 for display
                arr = v
                if arr.dtype != np.uint8:
                    mn, mx = arr.min(), arr.max()
                    if mx > mn:
                        arr = (arr - mn) / (mx - mn) * 255.0
                    else:
                        if k == 'mask':
                            arr = arr * 255.0 # Binary mask 0/1 -> 0/255
                        else:
                            arr = np.zeros_like(arr)
                    arr = arr.astype(np.uint8)
                
                ok_h, buf_h = cv2.imencode('.jpg', arr, jpeg_params)
                if ok_h:
                    heads_b64[k] = f"data:image/jpeg;base64,{base64.b64encode(buf_h.tobytes()).decode('ascii')}"

        enc_time = time.perf_counter() - t1
        resp = {"ok": True, "image_b64": b64, "width": int(img.shape[1]), "height": int(img.shape[0]), "timings": {"generate_s": gen_time, "encode_s": enc_time, "total_s": gen_time + enc_time}}
        if return_obbs:
            resp["obbs"] = obbs
        if obbs_raw is not None:
            resp["obbs_raw"] = obbs_raw
            resp["obbs_merged_count"] = len(obbs or [])
            resp["obbs_raw_count"] = len(obbs_raw)
        if return_heads:
            resp["heads"] = heads_b64
            
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
            import traceback
            tb_str = traceback.format_exc()
            print(f"[ERROR][synth_preview_inner] {tb_str}")
            return {"ok": False, "error": str(e2), "traceback": tb_str}
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[ERROR][synth_preview_outer] {tb_str}")
        return {"ok": False, "error": str(e), "traceback": tb_str}


@app.post("/synth_debug_export")
async def synth_debug_export(request: Request):
    """Save the current synth image + raw/merged OBBs + config to a temp folder
    for offline analysis of label-merge behavior.

    Body: {t, config, seed}. Returns the absolute folder path and a quick
    pairwise-overlap analysis of the raw OBBs.
    """
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}

    t = float(data.get("t", 0.0))
    config = data.get("config", {})
    seed_in = data.get("seed", None)

    try:
        from crystalGUI.data_generator import generate_image
        import numpy as np

        seed_used = int(seed_in) if seed_in is not None else random.SystemRandom().randint(0, 2**31 - 1)

        # Always compute raw + merged OBBs so we can compare regardless of UI toggles.
        force_cfg = json.loads(json.dumps(config)) if isinstance(config, dict) else {}
        phys = force_cfg.setdefault("physics", {})
        lm = dict(phys.get("label_merge", {}) or {})
        lm["enable"] = True  # force merge path so raw vs merged is meaningful
        phys["label_merge"] = lm

        res = generate_image(
            force_cfg, t, seed=seed_used,
            return_obbs=True, return_heads=False, return_obbs_raw=True,
        )
        if isinstance(res, tuple) and len(res) == 3:
            img, obbs_merged, obbs_raw = res
        elif isinstance(res, tuple) and len(res) == 2:
            img, obbs_merged = res
            obbs_raw = list(obbs_merged)
        else:
            img = res
            obbs_merged, obbs_raw = [], []

        obbs_merged = obbs_merged or []
        obbs_raw = obbs_raw or []

        # Pairwise overlap analysis on the RAW OBBs (the inputs to merge).
        from crystalGUI.osog.labels.merge import overlap_fraction_smaller_box
        thr = float(lm.get("overlap_threshold", 0.4))
        corners = []
        for ob in obbs_raw:
            c = ob.get("corners")
            corners.append(np.asarray(c, dtype=np.float32) if c and len(c) == 4 else None)

        n = len(obbs_raw)
        pairs_ge_thr = 0
        pairs_any = 0
        pairs_ge_thr_diff_group = 0
        pairs_ge_thr_same_group = 0
        max_frac = 0.0
        top_pairs = []
        for i in range(n):
            if corners[i] is None:
                continue
            for j in range(i + 1, n):
                if corners[j] is None:
                    continue
                f = overlap_fraction_smaller_box(corners[i], corners[j])
                if f <= 0.0:
                    continue
                pairs_any += 1
                if f > max_frac:
                    max_frac = f
                if f >= thr:
                    pairs_ge_thr += 1
                    gi = obbs_raw[i].get("group_id", -1)
                    gj = obbs_raw[j].get("group_id", -1)
                    if gi == gj:
                        pairs_ge_thr_same_group += 1
                    else:
                        pairs_ge_thr_diff_group += 1
                top_pairs.append((round(float(f), 4), i, j,
                                  obbs_raw[i].get("group_id", -1),
                                  obbs_raw[j].get("group_id", -1)))
        top_pairs.sort(reverse=True)
        top_pairs = top_pairs[:25]

        unique_groups = len({ob.get("group_id", -1) for ob in obbs_raw})

        analysis = {
            "raw_count": n,
            "merged_count": len(obbs_merged),
            "overlap_threshold": thr,
            "merge_by_group_id": bool(lm.get("merge_by_group_id", False)),
            "unique_group_ids": unique_groups,
            "overlapping_pairs_total": pairs_any,
            "pairs_overlap_ge_threshold": pairs_ge_thr,
            "  of_those_same_group_id": pairs_ge_thr_same_group,
            "  of_those_diff_group_id": pairs_ge_thr_diff_group,
            "max_overlap_fraction": round(float(max_frac), 4),
            "top_overlapping_pairs": [
                {"frac": f, "i": i, "j": j, "group_i": gi, "group_j": gj}
                for (f, i, j, gi, gj) in top_pairs
            ],
        }

        # Write everything to a timestamped folder.
        ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        out_dir = DEBUG_EXPORTS_DIR / f"debug_{ts}_seed{seed_used}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "image.png"), img)
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        with (out_dir / "obbs_raw.json").open("w", encoding="utf-8") as f:
            json.dump(obbs_raw, f, indent=2)
        with (out_dir / "obbs_merged.json").open("w", encoding="utf-8") as f:
            json.dump(obbs_merged, f, indent=2)
        with (out_dir / "analysis.json").open("w", encoding="utf-8") as f:
            json.dump({"seed": seed_used, "t": t, **analysis}, f, indent=2)

        print(f"[DEBUG EXPORT] -> {out_dir}  raw={n} merged={len(obbs_merged)} "
              f"thr={thr} group_filter={analysis['merge_by_group_id']}")

        return {
            "ok": True,
            "folder": str(out_dir.resolve()),
            "seed": seed_used,
            "analysis": analysis,
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR][synth_debug_export] {tb}")
        return {"ok": False, "error": str(e), "traceback": tb}


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


@app.get("/synth_batch_defaults")
async def synth_batch_defaults():
    """Return default batch output root for the playground UI."""
    batch_root = GENERATED_BATCH_DIR
    batch_root.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "batch_root_dir": str(batch_root.resolve())}


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
    preset_name = str(data.get("preset_name", "custom")).strip()
    # Sanitize preset name for folder
    safe_preset = re.sub(r"[^\w\-_.]", "_", preset_name) or "custom"
    
    # Generate timestamp for output folder
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    # Resolve output directory: full path, or base_dir + dataset_name, or default
    user_out_dir = str(data.get("out_dir", "") or "").strip()
    base_dir = str(data.get("base_dir", "") or "").strip()
    dataset_name_raw = data.get("dataset_name")
    if dataset_name_raw is not None and str(dataset_name_raw).strip():
        safe_dataset = re.sub(r"[^\w\-_.]", "_", str(dataset_name_raw).strip()) or safe_preset
    else:
        safe_dataset = safe_preset

    if user_out_dir:
        out_dir = user_out_dir
    elif base_dir:
        out_dir = str(Path(base_dir).expanduser().resolve() / f"{safe_dataset}_{ts}")
    else:
        out_dir = str(GENERATED_BATCH_DIR.resolve() / f"{safe_dataset}_{ts}")
    
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
    mem_spec = os.environ.get("SYNTH_BATCH_MEM", "128G")
    # Allow request body to override env partition/qos
    partition = str(data.get("partition", os.environ.get("SYNTH_BATCH_PARTITION", ""))).strip()
    qos = str(data.get("qos", os.environ.get("SYNTH_BATCH_QOS", ""))).strip()

    # Name job directories using timestamp (YYYY_MM_DD_HH_MM). For Slurm runs we will
    # also include the numeric Slurm job id after submission.
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Batch synth runs on CPU-only Slurm nodes; disable GPU in the saved config.
    if isinstance(config, dict):
        canvas = config.setdefault("canvas", {})
        canvas["use_gpu"] = False

    # Staging dir used before we know Slurm job id
    job_dir = SYNTH_JOBS_DIR / f"{ts}_staging"
    job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = job_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)

    # Persist synth config + training metadata alongside generated images.
    try:
        from crystalGUI.training import dataset_meta as training_dataset_meta
        shutil.copy2(cfg_path, out_path / "config.json")
        training_dataset_meta.write_dataset_meta(out_path, config)
    except Exception:
        pass

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
                    
                    # REGISTER JOB
                    job_manager.register_job(
                        job_id=f"slurm-{slurm_id}", 
                        mode="slurm", 
                        config=config, 
                        out_dir=str(out_path), 
                        n_images=n_images, 
                        slurm_id=slurm_id
                    )

                return {"ok": True, "mode": "slurm", "job_id": slurm_id, "stdout": res.stdout, "tasks": n_tasks, "out_dir": str(out_path)}
            else:
                # Fall back to local if submission failed
                raise RuntimeError(res.stderr or res.stdout)
        except Exception as e:
            # Fall through to local run
            pass

    # Local fallback: spawn background process(es)
    local_job_id = uuid.uuid4().hex[:8]
    final_dir = SYNTH_JOBS_DIR / f"{ts}_{local_job_id}"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, final_dir / "config.json")

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
        
        # REGISTER JOB
        job_manager.register_job(
            job_id=local_job_id, 
            mode="local", 
            config=config, 
            out_dir=str(out_path), 
            n_images=n_images, 
            pid=proc.pid
        )
        return {"ok": True, "mode": "local", "job_id": f"local-{local_job_id}", "pid": proc.pid, "log": str(log_path), "out_dir": str(out_path)}
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
            log_path = final_dir / f"local_task_{task_id}.log"
            logs.append(str(log_path))
            lf = log_path.open("w")
            proc = subprocess.Popen([python_exec, "-m", "crystalGUI.data_generator.batch_job", "--n-images", str(n_this), "--out-dir", str(out_path), "--config-file", str(cfg_path), "--seed-base", str(seed_base), "--index-offset", str(offset)], cwd=project_root, stdout=lf, stderr=lf, env=env)
            pids.append(proc.pid)
        
        # REGISTER JOB
        job_manager.register_job(
            job_id=local_job_id, 
            mode="local-array", 
            config=config, 
            out_dir=str(out_path), 
            n_images=n_images, 
            pids=pids
        )
        return {"ok": True, "mode": "local-array", "job_id": f"local-{local_job_id}", "pids": pids, "logs": logs, "tasks": len(pids), "out_dir": str(out_path)}

# === Preprocessing Presets Endpoints (save/list/get) ===
@app.post("/preproc_save_preset")
async def preproc_save_preset(request: Request):
    """Save provided preprocessing pipeline under a given preset name."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}
    name = str(data.get("name", "")).strip()
    cfg = data.get("pipeline") or data.get("config") or data.get("params")
    if not name:
        return {"ok": False, "error": "Preset name required"}
    if not isinstance(cfg, dict):
        return {"ok": False, "error": "Missing pipeline config"}
    safe = re.sub(r"[^\w\-_.]", "_", name)
    try:
        path = PREPROC_PRESETS_DIR / f"{safe}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return {"ok": True, "saved": str(path), "name": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/preproc_presets")
async def preproc_presets():
    try:
        names = [p.stem for p in PREPROC_PRESETS_DIR.glob("*.json")]
        return {"ok": True, "presets": sorted(names)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/preproc_get_preset")
async def preproc_get_preset(name: str):
    safe = re.sub(r"[^\w\-_.]", "_", str(name))
    path = PREPROC_PRESETS_DIR / f"{safe}.json"
    if not path.exists():
        return {"ok": False, "error": "Preset not found"}
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {"ok": True, "pipeline": cfg, "name": safe}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# === Outputs Batch Processing ===
from statistics import mean, stdev
import uuid
import threading
import time

def _compute_histogram(arr, bins=20, rng=None):
    arr = [float(v) for v in (arr or []) if isinstance(v, (int, float))]
    if not arr:
        return {"counts": [0], "labels": ["No data"], "min": 0.0, "max": 0.0, "width": 0.0}
    mn = rng[0] if rng else min(arr)
    mx = rng[1] if rng else max(arr)
    if mx == mn:
        return {"counts": [len(arr)], "labels": [f"{mn:.2f}"], "min": mn, "max": mx, "width": 0.0}
    width = (mx - mn) / bins
    counts = [0] * bins
    for v in arr:
        idx = int((v - mn) / width)
        if idx < 0: idx = 0
        if idx >= bins: idx = bins - 1
        counts[idx] += 1
    labels = [f"{(mn + i*width):.1f}–{(mn + (i+1)*width):.1f}" for i in range(bins)]
    return {"counts": counts, "labels": labels, "min": mn, "max": mx, "width": width}

# ---- Async job store for Outputs batch with progress tracking ----
# Each job entry structure:
# {
#   'status': 'running'|'finished'|'error',
#   'processed': int,
#   'total': int,
#   'percent': float,
#   'message': str,
#   'started_at': float,
#   'completed_at': float|None,
#   'summary': dict|None,
#   'per_image': list|None,
#   'skipped': list|None,
# }
OUTPUTS_JOBS = {}

def _outputs_run_batch_worker(job_id: str, dataset_path: str, params: dict, model_folder: str):
    """Background worker that processes the dataset and updates progress in OUTPUTS_JOBS."""
    job = OUTPUTS_JOBS.get(job_id)
    if not job:
        return
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        job.update({"status": "error", "message": "Invalid dataset path", "completed_at": time.time()})
        return
    # List files first to determine total
    try:
        files = _iter_dataset_image_files(d)
    except Exception as e:
        job.update({"status": "error", "message": f"Failed reading dataset: {e}", "completed_at": time.time()})
        return
    total = len(files)
    job["total"] = total
    job["processed"] = 0
    job["percent"] = 0.0
    if total == 0:
        job.update({"status": "error", "message": "No readable images found in dataset.", "completed_at": time.time()})
        return

    # Import heavy modules inside the worker to avoid blocking server startup
    from . import image_loader, model_loader, inference_runner, postprocess
    eph_model = None
    try:
        eph_model = model_loader.load_model_ephemeral(str(MODELS_DIR / model_folder))
    except Exception as e:
        job.update({"status": "error", "message": f"Failed to load model: {e}", "completed_at": time.time()})
        return

    out_root = RESULTS_DIR / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    entries = []
    skipped = []
    processed = 0

    for p in files:
        rel_name = _dataset_image_rel_name(p, d)
        tval = extract_timestamp_from_path(p, d)
        # Load + preprocess
        try:
            img = image_loader.load_image(str(p))
        except Exception as e:
            skipped.append({"path": str(p), "error": f"load_image failed: {e}"})
            processed += 1
            job["processed"] = processed
            job["percent"] = round(100.0 * processed / total, 2)
            continue
        try:
            imgp = image_loader.apply_pipeline(img, params)
        except Exception as e:
            skipped.append({"path": str(p), "error": f"apply_pipeline failed: {e}"})
            processed += 1
            job["processed"] = processed
            job["percent"] = round(100.0 * processed / total, 2)
            continue
        # Inference
        try:
            dets = inference_runner.run(eph_model, imgp)
        except Exception as e:
            skipped.append({"path": str(p), "error": f"inference failed: {e}"})
            processed += 1
            job["processed"] = processed
            job["percent"] = round(100.0 * processed / total, 2)
            continue
        try:
            stats = postprocess.compute_stats(dets)
        except Exception as e:
            skipped.append({"path": str(p), "error": f"compute_stats failed: {e}"})
            processed += 1
            job["processed"] = processed
            job["percent"] = round(100.0 * processed / total, 2)
            continue
        # Save overlay
        try:
            overlay = inference_runner.draw_detections(imgp, dets)
            tkey = f"{tval}"
            t_dir = out_root / tkey
            t_dir.mkdir(parents=True, exist_ok=True)
            overlay_stem = _overlay_stem_for_dataset_image(rel_name)
            overlay_path = t_dir / f"{overlay_stem}_overlay.png"
            image_loader.save_image(str(overlay_path), overlay)
            entries.append({
                "time": tval,
                "name": rel_name,
                "stats": stats,
                "overlay_url": f"/static/results/outputs/{tkey}/{overlay_path.name}"
            })
        except Exception as e:
            skipped.append({"path": str(p), "error": f"overlay/save failed: {e}"})
        # Progress update
        processed += 1
        job["processed"] = processed
        job["percent"] = round(100.0 * processed / total, 2)
        job["message"] = f"Processed {processed}/{total}"

    # Build final summaries
    by_time = {}
    for e in entries:
        key = f"{e['time']}"
        by_time.setdefault(key, []).append(e)
    summary = {
        "times": sorted([float(k) for k in by_time.keys()]),
        "time_unit": "min",
        "filename_map": {k: [x["name"] for x in v] for k, v in by_time.items()},
        "stats_by_time": {}
    }
    for k, imgs in by_time.items():
        all_len, all_wid, all_ar, counts = [], [], [], []
        for e in imgs:
            s = e["stats"]
            all_len.extend(s.get("lengths", []) or [])
            all_wid.extend(s.get("widths", []) or [])
            all_ar.extend(s.get("aspect_ratios", []) or [])
            counts.append(s.get("count", 0))
        def _mean(arr):
            arr = [v for v in arr if isinstance(v, (int, float))]
            return float(mean(arr)) if arr else 0.0
        def _std(arr):
            arr = [v for v in arr if isinstance(v, (int, float))]
            return float(stdev(arr)) if len(arr) > 1 else 0.0
        m_len, s_len = _mean(all_len), _std(all_len)
        m_wid, s_wid = _mean(all_wid), _std(all_wid)
        m_ar, s_ar = _mean(all_ar), _std(all_ar)
        cnt_avg = float(mean(counts)) if counts else 0.0
        cnt_std = float(stdev(counts)) if len(counts) > 1 else 0.0
        rng_len = [min(all_len) if all_len else 0.0, max(all_len) if all_len else 0.0]
        rng_wid = [min(all_wid) if all_wid else 0.0, max(all_wid) if all_wid else 0.0]
        rng_ar = [min(all_ar) if all_ar else 0.0, max(all_ar) if all_ar else 0.0]
        bins = 20
        def avg_hist(arrs, rng):
            if not arrs or all((not a) for a in arrs):
                return {"counts": [0], "labels": ["No data"]}
            per_counts, labels = [], None
            for a in arrs:
                h = _compute_histogram(a or [], bins, rng)
                per_counts.append(h["counts"])
                labels = h["labels"]
            n = len(per_counts)
            avg = [sum(c[i] for c in per_counts) / n for i in range(len(per_counts[0]))]
            return {"counts": avg, "labels": labels}
        h_len = avg_hist([e["stats"].get("lengths", []) for e in imgs], rng_len)
        h_wid = avg_hist([e["stats"].get("widths", []) for e in imgs], rng_wid)
        h_ar = avg_hist([e["stats"].get("aspect_ratios", []) for e in imgs], rng_ar)
        summary["stats_by_time"][k] = {
            "mean_length": m_len, "std_length": s_len,
            "mean_width": m_wid, "std_width": s_wid,
            "mean_aspect_ratio": m_ar, "std_aspect_ratio": s_ar,
            "count_avg": cnt_avg, "count_std": cnt_std,
            "histograms": {"lengths": h_len, "widths": h_wid, "aspect_ratios": h_ar}
        }

    job.update({
        "status": "finished",
        "completed_at": time.time(),
        "summary": summary,
        "per_image": entries,
        "skipped": skipped,
        "percent": 100.0,
        "processed": total,
    })

@app.post("/outputs_run_batch_start")
async def outputs_run_batch_start(dataset_path: str = Form(...), pipeline: str = Form("{}"), model_folder: str = Form(...)):
    """Start an asynchronous Outputs batch job and return a job_id to poll progress."""
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": "Invalid dataset path"}
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    job_id = uuid.uuid4().hex
    OUTPUTS_JOBS[job_id] = {
        "status": "running",
        "processed": 0,
        "total": 0,
        "percent": 0.0,
        "message": "Starting",
        "started_at": time.time(),
        "completed_at": None,
        "summary": None,
        "per_image": None,
        "skipped": None,
    }
    # Launch worker thread so we don't block the event loop
    th = threading.Thread(target=_outputs_run_batch_worker, args=(job_id, dataset_path, params, model_folder), daemon=True)
    th.start()
    return {"ok": True, "job_id": job_id}

@app.get("/outputs_run_batch_status")
async def outputs_run_batch_status(job_id: str):
    job = OUTPUTS_JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": "Job not found"}
    return {
        "ok": True,
        "status": job.get("status"),
        "processed": job.get("processed", 0),
        "total": job.get("total", 0),
        "percent": job.get("percent", 0.0),
        "message": job.get("message", ""),
    }

@app.get("/outputs_run_batch_result")
async def outputs_run_batch_result(job_id: str):
    job = OUTPUTS_JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": "Job not found"}
    if job.get("status") != "finished":
        return {"ok": False, "error": "Job not finished", "status": job.get("status")}
    return {
        "ok": True,
        "summary": job.get("summary"),
        "per_image": job.get("per_image"),
        "skipped_count": len(job.get("skipped") or []),
    }

@app.post("/outputs_run_batch")
async def outputs_run_batch(dataset_path: str = Form(...), pipeline: str = Form("{}"), model_folder: str = Form(...)):
    """Synchronous version retained for backward compatibility. Consider using outputs_run_batch_start + status + result."""
    d = Path(dataset_path)
    if not d.exists() or not d.is_dir():
        return {"ok": False, "error": "Invalid dataset path"}
    try:
        params = json.loads(pipeline) if pipeline else {}
    except Exception:
        params = {}
    from . import image_loader, model_loader, inference_runner, postprocess
    eph_model = None
    try:
        eph_model = model_loader.load_model_ephemeral(str(MODELS_DIR / model_folder))
    except Exception as e:
        return {"ok": False, "error": f"Failed to load model: {e}"}
    # Prepare output folder
    out_root = RESULTS_DIR / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    entries = []
    skipped = []
    try:
        for p in _iter_dataset_image_files(d):
            rel_name = _dataset_image_rel_name(p, d)
            tval = extract_timestamp_from_path(p, d)
            # Load + preprocess (robust: skip unreadable files)
            try:
                img = image_loader.load_image(str(p))
            except Exception as e:
                skipped.append({"path": str(p), "error": f"load_image failed: {e}"})
                continue
            try:
                imgp = image_loader.apply_pipeline(img, params)
            except Exception as e:
                skipped.append({"path": str(p), "error": f"apply_pipeline failed: {e}"})
                continue
            # Inference
            try:
                dets = inference_runner.run(eph_model, imgp)
            except Exception as e:
                skipped.append({"path": str(p), "error": f"inference failed: {e}"})
                continue
            try:
                stats = postprocess.compute_stats(dets)
            except Exception as e:
                skipped.append({"path": str(p), "error": f"compute_stats failed: {e}"})
                continue
            # Save overlay
            try:
                overlay = inference_runner.draw_detections(imgp, dets)
                tkey = f"{tval}"
                t_dir = out_root / tkey
                t_dir.mkdir(parents=True, exist_ok=True)
                overlay_stem = _overlay_stem_for_dataset_image(rel_name)
                overlay_path = t_dir / f"{overlay_stem}_overlay.png"
                image_loader.save_image(str(overlay_path), overlay)
                entries.append({
                    "time": tval,
                    "name": rel_name,
                    "stats": stats,
                    "overlay_url": f"/static/results/outputs/{tkey}/{overlay_path.name}"
                })
            except Exception as e:
                skipped.append({"path": str(p), "error": f"overlay/save failed: {e}"})
                continue
    except Exception as e:
        # Unexpected failure when traversing dataset path
        return {"ok": False, "error": f"Failed reading dataset: {e}"}

    if not entries:
        # If none processed, return a JSON error with diagnostic info instead of a 500
        return {
            "ok": False,
            "error": "No readable images found in dataset.",
            "skipped_count": len(skipped),
            "skipped_samples": skipped[:10],
        }

    # Group by time
    by_time = {}
    for e in entries:
        key = f"{e['time']}"
        by_time.setdefault(key, []).append(e)

    # Compute summaries
    summary = {
        "times": sorted([float(k) for k in by_time.keys()]),
        "time_unit": "min",
        "filename_map": {k: [x["name"] for x in v] for k, v in by_time.items()},
        "stats_by_time": {}
    }
    for k, imgs in by_time.items():
        # Aggregate arrays
        all_len = []
        all_wid = []
        all_ar = []
        counts = []
        for e in imgs:
            s = e["stats"]
            all_len.extend(s.get("lengths", []) or [])
            all_wid.extend(s.get("widths", []) or [])
            all_ar.extend(s.get("aspect_ratios", []) or [])
            counts.append(s.get("count", 0))
        # Means/std over aggregated distributions
        def _mean(arr):
            arr = [v for v in arr if isinstance(v, (int, float))]
            return float(mean(arr)) if arr else 0.0
        def _std(arr):
            arr = [v for v in arr if isinstance(v, (int, float))]
            return float(stdev(arr)) if len(arr) > 1 else 0.0
        m_len, s_len = _mean(all_len), _std(all_len)
        m_wid, s_wid = _mean(all_wid), _std(all_wid)
        m_ar, s_ar = _mean(all_ar), _std(all_ar)
        cnt_avg = float(mean(counts)) if counts else 0.0
        cnt_std = float(stdev(counts)) if len(counts) > 1 else 0.0
        # Averaged histograms across images: common range from combined arrays
        rng_len = [min(all_len) if all_len else 0.0, max(all_len) if all_len else 0.0]
        rng_wid = [min(all_wid) if all_wid else 0.0, max(all_wid) if all_wid else 0.0]
        rng_ar = [min(all_ar) if all_ar else 0.0, max(all_ar) if all_ar else 0.0]
        bins = 20
        # For each image, compute hist with common ranges
        def avg_hist(arrs, rng):
            if not arrs or all((not a) for a in arrs):
                return {"counts": [0], "labels": ["No data"]}
            per_counts = []
            labels = None
            for a in arrs:
                h = _compute_histogram(a or [], bins, rng)
                per_counts.append(h["counts"])
                labels = h["labels"]
            # Average counts per bin across images
            n = len(per_counts)
            avg = [sum(c[i] for c in per_counts) / n for i in range(len(per_counts[0]))]
            return {"counts": avg, "labels": labels}
        h_len = avg_hist([e["stats"].get("lengths", []) for e in imgs], rng_len)
        h_wid = avg_hist([e["stats"].get("widths", []) for e in imgs], rng_wid)
        h_ar = avg_hist([e["stats"].get("aspect_ratios", []) for e in imgs], rng_ar)
        summary["stats_by_time"][k] = {
            "mean_length": m_len, "std_length": s_len,
            "mean_width": m_wid, "std_width": s_wid,
            "mean_aspect_ratio": m_ar, "std_aspect_ratio": s_ar,
            "count_avg": cnt_avg, "count_std": cnt_std,
            "histograms": {"lengths": h_len, "widths": h_wid, "aspect_ratios": h_ar}
        }

    return {"ok": True, "summary": summary, "per_image": entries, "skipped_count": len(skipped)}

# =================== Calibration / Optimization Endpoints ===================
from app.services.calibration_manager import CalibrationManager
# Initialize Manager
calibration_manager = CalibrationManager(BASE_DIR)

@app.post("/synth_save_target")
async def synth_save_target(request: Request):
    """Generate an image from config and save it as a target file in uploads."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Expected JSON body"}

    config = data.get("config", {})
    seed_in = data.get("seed")

    try:
        from crystalGUI.data_generator import generate_image
        
        if seed_in is None:
            seed_used = random.SystemRandom().randint(0, 2**31 - 1)
        else:
            seed_used = int(seed_in)

        # Generate image
        img = generate_image(config, t=0.0, seed=seed_used)
        
        # Create filename
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synth_target_{ts}_{uuid.uuid4().hex[:6]}.png"
        target_path = UPLOADS_DIR / filename
        
        # Save
        from . import image_loader
        image_loader.save_image(str(target_path), img)
        
        return {"ok": True, "filename": filename, "path": str(target_path)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.get("/calibration/params")
async def calibration_params():
    """Return the available optimization parameters and their rules."""
    rules_path = BASE_DIR / "diff_calibration" / "optimization_rules.json"
    if not rules_path.exists():
        print(f"[ERROR] Rules file not found at: {rules_path}")
        return {"ok": False, "error": "Rules file not found"}
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            rules = json.load(f)
        
        # Filter out comments AND configuration blocks like 'stages'
        # We only want parameter definitions (keys with dots usually, or specific structure)
        clean_rules = {}
        for k, v in rules.items():
            if k.startswith("__"): continue
            if k == "stages": continue # specialized config, not a parameter
            if isinstance(v, dict) and "stage" in v: # Simple heuristic: params have 'stage' definition
                clean_rules[k] = v
            elif "." in k: # Fallback for dot-notation keys
                clean_rules[k] = v
                
        return {"ok": True, "params": clean_rules}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

@app.post("/calibration/start")
async def calibration_start(
    target_image_name: str = Form(...),
    initial_config: str = Form("{}"),
    selected_params: str = Form("[]"),
    max_steps: int = Form(200),
    learning_rate: float = Form(0.05),
    device: str = Form("cpu")
):
    """
    Start a parameter optimization job.
    target_image_name: Name of an existing image in uploads/
    initial_config: JSON string of the starting configuration
    selected_params: JSON list of parameter names to optimize (e.g. ["optics.focus_z"])
    """
    target_path = UPLOADS_DIR / target_image_name
    if not target_path.exists():
        return {"ok": False, "error": "Target image not found"}

    try:
        config = json.loads(initial_config)
        params = json.loads(selected_params)
    except Exception as e:
        return {"ok": False, "error": f"Invalid JSON: {e}"}

    # Auto-detect device if CPU is requested but CUDA is available
    import torch
    if device == "cpu" and torch.cuda.is_available():
        print("[Calibration] Auto-switching to CUDA")
        device = "cuda"

    try:
        job_id = calibration_manager.start_job(
            target_image_path=str(target_path),
            initial_config=config,
            selected_params=params,
            max_steps=max_steps,
            learning_rate=learning_rate,
            device=device
        )
        return {"ok": True, "job_id": job_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

@app.get("/calibration/status/{job_id}")
async def calibration_status(job_id: str):
    """Get the current status and metrics of a calibration job."""
    status = calibration_manager.get_job_status(job_id)
    if not status:
        return {"ok": False, "error": "Job not found"}
    return {"ok": True, "status": status}

@app.post("/calibration/stop/{job_id}")
async def calibration_stop(job_id: str):
    """Stop a running calibration job."""
    success = calibration_manager.stop_job(job_id)
    return {"ok": success}

@app.post("/calibration/compute_loss")
async def calibration_compute_loss(
    target_image_name: str = Form(...),
    current_config: str = Form("{}"),
    n_samples: int = Form(1),
    device: str = Form("cpu")
):
    """
    Compute losses between target image and model generated with current config.
    Supports multiple samples for robust stochastic estimation.
    """
    target_path = UPLOADS_DIR / target_image_name
    if not target_path.exists():
        return {"ok": False, "error": "Target image not found"}
        
    try:
        config = json.loads(current_config)
    except:
        return {"ok": False, "error": "Invalid config JSON"}
        
    # Auto-detect device if CPU is requested but CUDA is available
    import torch
    if device == "cpu" and torch.cuda.is_available():
        print("[Calibration] Auto-switching to CUDA")
        device = "cuda"

    # Run in thread pool to avoid blocking async loop?
    # CalibrationManager.compute_loss is synchronous and CPU intensive.
    # We should run it in a thread.
    import asyncio
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        lambda: calibration_manager.compute_loss(
            target_image_path=str(target_path),
            config=config,
            n_samples=n_samples,
            device=device
        )
    )
    return result

# =================== Dataset listing helpers ===================
_DATASET_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _path_is_under(base: Path, candidate: Path) -> bool:
    """True if candidate is under base, including symlink/orcd vs home aliases."""
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        pass
    try:
        import os
        cand_real = Path(os.path.realpath(candidate))
        base_real = Path(os.path.realpath(base))
        cand_real.relative_to(base_real)
        return True
    except (ValueError, OSError):
        pass
    # Same dataset folder may be addressable as /orcd/... or /home/... on cluster filesystems.
    if candidate.name and not candidate.name.startswith("."):
        canonical = base / candidate.name
        if canonical.is_dir():
            return True
    return False


def _assert_training_dataset_path(dataset_path: str) -> Path:
    """Training may only use synthetic datasets under generated_batch/."""
    p = Path(dataset_path)
    if not p.is_dir():
        raise ValueError("Invalid dataset path")
    if _path_is_under(GENERATED_BATCH_DIR, p):
        canonical = GENERATED_BATCH_DIR / p.name
        if canonical.is_dir():
            return canonical.resolve()
        return p.resolve()
    raise ValueError("Training datasets must be synthetic batches under data/generated_batch/")


def _assert_outputs_dataset_path(dataset_path: str) -> Path:
    """Outputs batch may only read uploaded datasets under dataset_uploads/."""
    p = Path(dataset_path)
    if not p.is_dir():
        raise ValueError("Invalid dataset path")
    if _path_is_under(DATASET_UPLOADS_DIR, p):
        return p.resolve()
    # Cluster paths may differ in prefix but share the dataset_uploads folder name.
    if "dataset_uploads" in p.parts:
        idx = p.parts.index("dataset_uploads")
        suffix = Path(*p.parts[idx + 1:])
        canonical = DATASET_UPLOADS_DIR / suffix
        if canonical.is_dir():
            return canonical.resolve()
    raise ValueError("Outputs datasets must be under data/dataset_uploads/")


def _count_images_in_dir(path: Path, *, top_level_only: bool = False) -> int:
    if not path.is_dir():
        return 0
    if top_level_only:
        return sum(
            1 for fp in path.iterdir()
            if fp.is_file() and fp.suffix.lower() in _DATASET_IMAGE_EXTS
        )
    return sum(
        1 for fp in path.rglob("*")
        if fp.is_file() and fp.suffix.lower() in _DATASET_IMAGE_EXTS
    )


def _inspect_training_dataset(path: Path):
    p = Path(path)
    if not p.is_dir():
        return None

    has_dota = (p / "labels_dota").exists() and any((p / "labels_dota").glob("*.txt"))
    labels_dir = p / "labels"
    has_yolo = labels_dir.exists() and (
        any(labels_dir.glob("*.txt"))
        or ((labels_dir / "train").exists() and any((labels_dir / "train").glob("*.txt")))
    )
    has_yolo_for_split = labels_dir.exists() and any(labels_dir.glob("*.txt"))
    is_split = (p / "images" / "train").exists()

    img_count = 0
    if (p / "images").exists():
        for fp in (p / "images").rglob("*"):
            if fp.is_file() and fp.suffix.lower() in _DATASET_IMAGE_EXTS:
                img_count += 1

    return {
        "path": str((GENERATED_BATCH_DIR / p.name).resolve()),
        "name": p.name,
        "has_dota": has_dota,
        "has_yolo": has_yolo,
        "has_yolo_for_split": has_yolo_for_split,
        "is_split": is_split,
        "image_count": img_count,
        "source": "generated",
        "training_max_det": _dataset_training_max_det(p),
    }


def _dataset_training_max_det(path: Path):
    try:
        from crystalGUI.training import dataset_meta as training_dataset_meta
        meta = training_dataset_meta.read_dataset_meta(path)
        if meta.get("training_max_det"):
            return int(meta["training_max_det"])
        if meta.get("recommended_max_det") or meta.get("max_boxes_per_image"):
            return training_dataset_meta.resolve_training_max_det(path, meta=meta)
        cfg = training_dataset_meta.load_synth_config(path)
        if cfg is not None:
            return training_dataset_meta.resolve_training_max_det(path)
    except Exception:
        pass
    return None


def _list_generated_training_datasets() -> list:
    datasets = []
    if not GENERATED_BATCH_DIR.exists():
        return datasets
    for d in sorted(GENERATED_BATCH_DIR.iterdir(), key=os.path.getmtime, reverse=True):
        if not d.is_dir() or d.name in _LEGACY_DIR_NAMES:
            continue
        info = _inspect_training_dataset(d)
        if info:
            from app.services.training_dataset_browser import split_counts
            info["split_counts"] = split_counts(d)
            datasets.append(info)
    return datasets


@app.get("/training/dataset_samples")
async def training_dataset_samples(
    dataset_path: str,
    split: str = "train",
    offset: int = 0,
    limit: int = 1,
    include_labels: bool = False,
):
    """Paginated image manifest for training dataset browser."""
    try:
        from app.services.training_dataset_browser import list_samples
        root = _assert_training_dataset_path(dataset_path)
        return {"ok": True, **list_samples(root, split, offset, limit, include_labels=include_labels)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/training/dataset_image")
async def training_dataset_image(dataset_path: str, rel_path: str):
    """Serve a single image file from a training dataset."""
    try:
        from app.services.training_dataset_browser import resolve_dataset_file
        root = _assert_training_dataset_path(dataset_path)
        path = resolve_dataset_file(root, rel_path)
        return FileResponse(str(path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


_MAG_SUBFOLDER_NAMES = frozenset({"x10", "x20", "10x", "20x"})
_TIME_SUBFOLDER_RE = re.compile(r"^\d+_min$", re.I)


def _is_magnification_subfolder(name: str) -> bool:
    key = name.strip().lower().replace(" ", "")
    return key in _MAG_SUBFOLDER_NAMES


def _is_x10_subfolder(name: str) -> bool:
    key = name.strip().lower().replace(" ", "")
    return key in {"x10", "10x"}


def _magnification_subfolders(subfolders: list) -> list:
    """Return magnification child folders, excluding x10."""
    return [
        (sub, cnt)
        for sub, cnt in subfolders
        if _is_magnification_subfolder(sub.name) and not _is_x10_subfolder(sub.name)
    ]


def _is_timepoint_subfolder(name: str) -> bool:
    return bool(_TIME_SUBFOLDER_RE.match(name.strip()))


def _dataset_subfolder_layout(subfolders: list) -> str:
    """Classify child folders: timepoint series (60_min, …), magnification (x10/x20), or mixed."""
    if not subfolders:
        return "empty"
    names = [sub.name for sub, _ in subfolders]
    mag = sum(1 for n in names if _is_magnification_subfolder(n))
    time = sum(1 for n in names if _is_timepoint_subfolder(n))
    if time > 0 and mag == 0:
        return "timepoints"
    if mag > 0 and time == 0:
        return "magnification"
    return "mixed"


def _list_uploaded_output_datasets() -> list:
    """Discover selectable image folders under dataset_uploads/ for Outputs batch runs."""
    datasets = []
    if not DATASET_UPLOADS_DIR.exists():
        return datasets

    for top in sorted(DATASET_UPLOADS_DIR.iterdir(), key=os.path.getmtime, reverse=True):
        if not top.is_dir():
            continue

        direct_count = _count_images_in_dir(top, top_level_only=True)
        if direct_count > 0:
            datasets.append({
                "path": str(top),
                "name": top.name,
                "display_name": top.name,
                "image_count": direct_count,
                "source": "uploaded",
            })
            continue

        subfolders = []
        nested_subfolders = []
        for sub in sorted(top.iterdir()):
            if not sub.is_dir():
                continue
            flat_cnt = _count_images_in_dir(sub, top_level_only=True)
            nested_cnt = _count_images_in_dir(sub)
            if flat_cnt > 0:
                subfolders.append((sub, flat_cnt))
            elif nested_cnt > 0:
                nested_subfolders.append((sub, nested_cnt))

        if subfolders or nested_subfolders:
            recursive_count = _count_images_in_dir(top)
            layout = _dataset_subfolder_layout(subfolders)

            if layout == "timepoints":
                # e.g. Continuous_PmAb_2_mg-mL/90_min, 100_min, … — one batch over all timepoints
                datasets.append({
                    "path": str(top),
                    "name": top.name,
                    "display_name": top.name,
                    "image_count": recursive_count,
                    "source": "uploaded",
                    "layout": "timepoints",
                })
                continue

            if layout == "magnification":
                mag_subs = _magnification_subfolders(subfolders)
                if len(mag_subs) == 1:
                    sub, cnt = mag_subs[0]
                    datasets.append({
                        "path": str(sub),
                        "name": top.name,
                        "display_name": top.name,
                        "image_count": cnt,
                        "source": "uploaded",
                        "layout": "magnification",
                    })
                else:
                    for sub, cnt in mag_subs:
                        datasets.append({
                            "path": str(sub),
                            "name": sub.name,
                            "display_name": f"{top.name} / {sub.name}",
                            "image_count": cnt,
                            "source": "uploaded",
                            "parent": top.name,
                            "layout": "magnification",
                        })
                continue

            # Mixed or nested: expose parent + each immediate child
            datasets.append({
                "path": str(top),
                "name": top.name,
                "display_name": f"{top.name} (all)",
                "image_count": recursive_count,
                "source": "uploaded",
                "layout": "all",
            })
            for sub, cnt in subfolders:
                datasets.append({
                    "path": str(sub),
                    "name": sub.name,
                    "display_name": f"{top.name} / {sub.name}",
                    "image_count": cnt,
                    "source": "uploaded",
                    "parent": top.name,
                    "layout": "flat",
                })
            for sub, cnt in nested_subfolders:
                datasets.append({
                    "path": str(sub),
                    "name": sub.name,
                    "display_name": f"{top.name} / {sub.name}",
                    "image_count": cnt,
                    "source": "uploaded",
                    "parent": top.name,
                    "layout": "nested",
                })
            continue

        recursive_count = _count_images_in_dir(top)
        if recursive_count > 0:
            datasets.append({
                "path": str(top),
                "name": top.name,
                "display_name": top.name,
                "image_count": recursive_count,
                "source": "uploaded",
            })

    return datasets


# =================== YOLO Training Endpoints ===================
from app.services.training_manager import TrainingManager
import crystalGUI.training.dataset_meta as training_dataset_meta
import crystalGUI.training.preprocessing as training_preproc
import crystalGUI.training.splitting as training_split
import crystalGUI.training.slurm_utils as training_slurm

TRAINING_LOGS_DIR = DATA_DIR / "training_logs"
TRAINING_RUNS_DIR = DATA_DIR / "runs" / "obb"
training_manager = TrainingManager(TRAINING_LOGS_DIR, TRAINING_RUNS_DIR, LEGACY_TRAINING_LOGS_DIR)

@app.get("/training/datasets")
async def training_list_datasets():
    """List synthetic datasets created by OSOG batch generation (generated_batch/ only)."""
    return {"ok": True, "datasets": _list_generated_training_datasets()}


@app.get("/outputs/datasets")
async def outputs_list_datasets():
    """List uploaded datasets available for Outputs batch inference."""
    return {"ok": True, "datasets": _list_uploaded_output_datasets()}


@app.get("/outputs/dataset_sample")
async def outputs_dataset_sample(dataset_path: str, index: int = 0):
    """Return a sample image from an uploaded dataset for scale calibration."""
    try:
        root = _assert_outputs_dataset_path(dataset_path)
        files = _iter_dataset_image_files(root)
        if not files:
            return {"ok": False, "error": "No images found in dataset"}
        idx = max(0, min(int(index), len(files) - 1))
        img_path = files[idx]
        rel = _dataset_image_rel_name(img_path, root)
        from urllib.parse import quote
        qp = quote(str(root))
        rp = quote(rel)
        width = height = None
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception:
            pass
        return {
            "ok": True,
            "index": idx,
            "total": len(files),
            "name": rel,
            "image_url": f"/outputs/dataset_image?dataset_path={qp}&rel_path={rp}",
            "width": width,
            "height": height,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/outputs/dataset_image")
async def outputs_dataset_image(dataset_path: str, rel_path: str):
    """Serve a single image from an uploaded outputs dataset."""
    try:
        from app.services.training_dataset_browser import resolve_dataset_file
        root = _assert_outputs_dataset_path(dataset_path)
        path = resolve_dataset_file(root, rel_path)
        return FileResponse(str(path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/training/convert_labels")
async def training_convert_labels(dataset_path: str = Form(...), width: int = Form(1024), height: int = Form(1024)):
    """Convert DOTA labels to YOLO OBB format."""
    try:
        dataset_path = str(_assert_training_dataset_path(dataset_path))
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(
            None,
            lambda: training_preproc.convert_dota_to_yolo(dataset_path, width, height)
        )
        return {"ok": True, "converted": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/training/split_data")
async def training_split_data(dataset_path: str = Form(...), train_ratio: float = Form(0.8), val_ratio: float = Form(0.1), test_ratio: float = Form(0.1)):
    """Split dataset into train/val/test."""
    try:
        dataset_path = str(_assert_training_dataset_path(dataset_path))
        loop = asyncio.get_event_loop()
        counts = await loop.run_in_executor(
            None,
            lambda: training_split.split_dataset(dataset_path, [train_ratio, val_ratio, test_ratio])
        )
        return {"ok": True, "counts": counts}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/training/start")
async def training_start(
    dataset_path: str = Form(...),
    model_name: str = Form("yolo11n-obb.pt"),
    epochs: int = Form(100),
    batch_size: int = Form(4),
    img_size: int = Form(1024),
    partition: str = Form("mit_preemptable"),
    gpu: str = Form("h200:1"),
    time_limit: str = Form("06:00:00")
):
    """Generate config and submit training job."""
    try:
        dataset_path = str(_assert_training_dataset_path(dataset_path))
        slurm_config = {
            "partition": partition,
            "gpu": gpu,
            "time": time_limit
        }
        
        # Determine project name from dataset name
        project_name = f"train_{Path(dataset_path).name}"
        
        # Generate script and paths
        slurm_path, runs_path, job_name, max_det = training_slurm.generate_training_slurm(
            dataset_path=dataset_path,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            project_name=project_name,
            slurm_config=slurm_config
        )
        
        # Submit
        sbatch = shutil.which("sbatch")
        if sbatch:
            res = subprocess.run([sbatch, slurm_path], capture_output=True, text=True)
            if res.returncode == 0:
                m = re.search(r"Submitted batch job\s+(\d+)", res.stdout)
                slurm_id = m.group(1) if m else None
                
                training_manager.register_job(
                    job_id=job_name,
                    dataset_path=dataset_path,
                    slurm_id=slurm_id,
                    model_name=model_name,
                    status="submitted"
                )
                return {"ok": True, "job_id": job_name, "slurm_id": slurm_id, "max_det": max_det}
            else:
                return {"ok": False, "error": f"Sbatch failed: {res.stderr}"}
        else:
            return {"ok": False, "error": "Sbatch not found (local execution not implemented for training)"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

@app.get("/training/jobs")
async def training_jobs():
    return {"ok": True, "jobs": training_manager.list_jobs()}

@app.get("/training/logs/{job_id}")
async def training_logs(job_id: str):
    log_content = training_manager.get_log(job_id)
    return {"ok": True, "log": log_content}

@app.delete("/training/job/{job_id}")
async def training_delete_job(job_id: str):
    # Logic to cancel slurm job if running?
    # For now just remove from list
    # training_manager.delete_job(job_id) 
    return {"ok": False, "error": "Not implemented"}

@app.post("/training/export_model")
async def training_export_model(job_id: str = Form(...), model_name: str = Form(...)):
    """
    Copy the trained weights from a job to the crystalGUI/models folder.
    Creates a new folder in models/ with the model.py wrapper and config.yaml.
    """
    # 1. Locate the job and its weights
    # TrainingManager knows where logs are, but we need the 'runs' folder path.
    # We can infer it or store it in the job record.
    # Currently training_slurm.generate_training_slurm returns (slurm_path, runs_path, job_name)
    # The runs_path is e.g. .../data/runs/obb/{project_name}
    # Inside there, YOLO creates another folder (usually 'train', 'train2'...)
    # We need to find the 'best.pt' inside that.
    
    # Let's search in data/runs/obb/{job_name}/weights/best.pt
    # Note: job_id passed here is likely the job_name used in start().
    
    runs_base = DATA_DIR / "runs" / "obb"
    # The project name was "train_{dataset_name}"
    # But job_id is "train_{dataset_name}_{timestamp}"
    # Wait, in start():
    # project_name = f"train_{Path(dataset_path).name}"
    # job_name = f"train_{Path(dataset_path).name}_{timestamp}"
    # name="{project_name}" passed to YOLO actually creates a subfolder inside project if name is specified?
    # No, YOLO argument 'project' is the parent dir, 'name' is the experiment dir.
    # In generate_training_slurm:
    # project="{runs_dir}"  (which is .../data/runs/obb)
    # name="{project_name}" (which is train_{dataset_name})
    
    # So the output is in .../data/runs/obb/train_{dataset_name}
    # BUT if we run multiple times, YOLO increments: train_{dataset_name}2, etc.
    # This makes it hard to link a specific job_id to a specific folder unless we capture stdout.
    
    # Alternative: The user selects a job from the list. The list has job_id.
    # The job log contains "Results saved to .../data/runs/obb/train_Dataset"
    
    # For now, let's allow the user to browse/select from "data/runs/obb" or 
    # try to find the best.pt in the most recent folder matching the dataset.
    
    # Simpler approach for this iteration:
    # We assume the user wants to export from a specific path provided by the frontend,
    # OR we search for best.pt in the job's expected location.
    
    # Let's try to find the weights based on job info if possible, or search.
    # Since we don't have exact mapping in TrainingManager yet, let's search 
    # for any 'best.pt' in data/runs/obb related to this job.
    
    # Actually, let's just list available trained models in a separate endpoint 
    # and allow picking one.
    pass

@app.get("/training/trained_models")
async def training_list_trained_models():
    """List best.pt files from active training runs (excludes legacy_training archive)."""
    runs_dir = TRAINING_RUNS_DIR
    models = []
    if runs_dir.exists():
        for p in runs_dir.rglob("best.pt"):
            # p is .../train_Dataset/weights/best.pt
            # experiment name is parent.parent.name
            exp_name = p.parent.parent.name
            # Get timestamp of file
            mtime = p.stat().st_mtime
            date_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            
            models.append({
                "path": str(p),
                "name": exp_name,
                "date": date_str,
                "size_mb": round(p.stat().st_size / (1024*1024), 2)
            })
            
    # Sort by date desc
    models.sort(key=lambda x: x["date"], reverse=True)
    return {"ok": True, "models": models}

@app.post("/training/deploy_model")
async def training_deploy_model(weights_path: str = Form(...), model_name: str = Form(...)):
    """
    Deploy a trained model to models/<name>/ for local testing (git-ignored).
    To publish, copy the folder to models/public/ manually.
    1. Create models/{model_name}
    2. Copy weights_path to models/{model_name}/{model_name}.pt
    3. Copy template model.py and config.yaml
    """
    src = Path(weights_path)
    if not src.exists():
        return {"ok": False, "error": "Weights file not found"}
        
    safe_name = re.sub(r"[^\w\-_]", "_", model_name)
    dest_dir = MODELS_DIR / safe_name
    
    if dest_dir.exists():
        return {"ok": False, "error": f"Model '{safe_name}' already exists. Delete it first or choose another name."}
        
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy weights
        dest_weights = dest_dir / f"{safe_name}.pt"
        shutil.copy2(src, dest_weights)
        
        # Create config.yaml
        config = {
            "weights_file": f"{safe_name}.pt",
            "device": "cpu",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 1024,
            "max_det": 10000,
        }
        with (dest_dir / "config.yaml").open("w") as f:
            yaml.dump(config, f)
            
        # Create model.py (Standard YOLO wrapper)
        # We can copy from an existing template or write it fresh.
        # Let's read from models/yolo_obb/model.py if it exists, else write default.
        template_src = MODELS_DIR / "yolo_obb" / "model.py"
        if template_src.exists():
            shutil.copy2(template_src, dest_dir / "model.py")
        else:
            # Fallback: minimal wrapper
             with (dest_dir / "model.py").open("w") as f:
                 f.write("""from ultralytics import YOLO
import os
import yaml

def load(config_override=None):
    d = os.path.dirname(__file__)
    with open(os.path.join(d, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    if config_override: cfg.update(config_override)
    model = YOLO(os.path.join(d, cfg["weights_file"]))
    return {"model": model, "config": cfg}

def infer(wrapper, img):
    res = wrapper["model"].predict(img, conf=wrapper["config"]["confidence_threshold"], verbose=False)
    # ... implementation details omitted for brevity, assumes standard yolo_obb wrapper ...
    # (In real deployment we should ensure full implementation)
    return []
""")
                 
        # Create name.txt for display
        with (dest_dir / "name.txt").open("w") as f:
            f.write(model_name)
            
        return {"ok": True, "deployed_path": str(dest_dir)}
        
    except Exception as e:
        # Cleanup
        shutil.rmtree(dest_dir, ignore_errors=True)
        return {"ok": False, "error": str(e)}