import importlib.util
from pathlib import Path
from typing import Optional, Dict

_current_model: Optional[Dict] = None


def _blob_default() -> Dict:
    try:
        import cv2
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 100000
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        detector = cv2.SimpleBlobDetector_create(params)
        return {"type": "blob", "detector": detector}
    except Exception as e:
        return {"type": "none", "error": f"Blob detector unavailable: {e}"}


def set_current_model(name_or_path: str):
    """Set current model by name ('blob', 'yolo') or by folder path containing model.py."""
    global _current_model
    val = name_or_path.strip()
    # If it's a directory, attempt plugin load
    p = Path(val)
    if p.exists() and p.is_dir():
        model_py = p / "model.py"
        if not model_py.exists():
            _current_model = {"type": "plugin", "error": f"model.py not found in {p}"}
            return
        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{p.name}", str(model_py))
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            # Expect plugin to define load() and infer(model, img)
            plugin_model = None
            if hasattr(mod, "load"):
                plugin_model = mod.load()
            _current_model = {"type": "plugin", "module": mod, "model": plugin_model, "path": str(p)}
        except Exception as e:
            _current_model = {"type": "plugin", "error": f"Failed to load plugin: {e}", "path": str(p)}
        return

    name_l = val.lower()
    if name_l in ("blob", "simple"):
        _current_model = _blob_default()
        return
    if name_l.startswith("yolo"):
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO("models/yolo.pt")
            _current_model = {"type": "yolo", "model": model}
        except Exception as e:
            _current_model = _blob_default()
            _current_model["fallback_error"] = str(e)
        return

    # default
    _current_model = _blob_default()


def get_current_model() -> Dict:
    global _current_model
    if _current_model is None:
        set_current_model("blob")
    return _current_model


def get_model_info() -> Dict:
    m = get_current_model()
    info = {"type": m.get("type")}
    if "path" in m:
        info["path"] = m["path"]
    if "fallback_error" in m:
        info["fallback_error"] = m["fallback_error"]
    if "error" in m:
        info["error"] = m["error"]
    return info