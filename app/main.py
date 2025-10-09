from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
from pathlib import Path
import shutil
import json
import uuid
import re
import asyncio

# Lazy import heavy modules inside endpoints to avoid failing startup


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
PREPROC_DIR = RESULTS_DIR / "preprocessed"
STREAM_DIR = RESULTS_DIR / "stream"

for p in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, PREPROC_DIR, STREAM_DIR]:
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Crystal Analysis GUI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    target = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
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
    return {"ok": True, "processed_path": f"/static/results/{out_path.name}"}


@app.post("/load_model")
async def load_model(name: str = Form(...)):
    from . import model_loader
    model_loader.set_current_model(name)
    model_info = model_loader.get_model_info()
    return {"ok": True, "model": model_info}


@app.post("/select_model_folder")
async def select_model_folder(folder_path: str = Form(...)):
    from . import model_loader
    model_loader.set_current_model(folder_path)
    return {"ok": True, "model": model_loader.get_model_info()}


@app.post("/inference")
async def run_inference(image_name: str = Form(...)):
    img_path = UPLOADS_DIR / image_name
    if not img_path.exists():
        return {"ok": False, "error": "Image not found"}
    from . import image_loader, model_loader, inference_runner, postprocess
    img = image_loader.load_image(str(img_path))
    model = model_loader.get_current_model()
    dets = inference_runner.run(model, img)
    stats = postprocess.compute_stats(dets)

    overlay = inference_runner.draw_detections(img, dets)
    overlay_path = RESULTS_DIR / f"{Path(image_name).stem}_overlay.png"
    image_loader.save_image(str(overlay_path), overlay)

    result = {
        "image": image_name,
        "detections": dets,
        "stats": stats,
        "overlay_url": f"/static/results/{overlay_path.name}",
    }
    result_path = RESULTS_DIR / f"{Path(image_name).stem}_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)

    return {"ok": True, **result}


# Serve results images via static mount alias
app.mount("/static/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


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


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    LIVE_CLIENTS.add(websocket)
    try:
        # Send current state immediately if present
        if LIVE_STATE.get("last"):
            await websocket.send_text(json.dumps({"ok": True, "last": LIVE_STATE["last"]}))
        # Keep connection alive with a simple heartbeat
        while True:
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        LIVE_CLIENTS.discard(websocket)