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
        return {"type": "blob", "name": "Simple Blob Detector", "detector": detector}
    except Exception as e:
        return {"type": "none", "name": "No Model", "error": f"Blob detector unavailable: {e}"}


def _load_model(name_or_path: str) -> Dict:
    """Load a model by name or folder path but DO NOT set global state. Returns a model dict."""
    val = name_or_path.strip()
    p = Path(val)
    if p.exists() and p.is_dir():
        model_py = p / "model.py"
        if not model_py.exists():
            return {"type": "plugin", "error": f"model.py not found in {p}"}
        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{p.name}", str(model_py))
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            plugin_model = None
            if hasattr(mod, "load"):
                plugin_model = mod.load()
            name_file = p / "name.txt"
            if name_file.exists():
                try:
                    display_name = name_file.read_text().strip()
                except:
                    display_name = p.name.replace('_', ' ').title()
            else:
                display_name = p.name.replace('_', ' ').title()
            return {"type": "plugin", "name": display_name, "module": mod, "model": plugin_model, "path": str(p)}
        except Exception as e:
            return {"type": "plugin", "name": f"Plugin Error ({p.name})", "error": f"Failed to load plugin: {e}", "path": str(p)}
    name_l = val.lower()
    if name_l in ("blob", "simple"):
        return _blob_default()
    if name_l.startswith("yolo"):
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO("models/yolo.pt")
            return {"type": "yolo", "name": "YOLO Detector", "model": model}
        except Exception as e:
            d = _blob_default()
            d["fallback_error"] = str(e)
            return d
    return _blob_default()


def set_current_model(name_or_path: str):
    """Set current model by name ('blob', 'yolo') or by folder path containing model.py."""
    global _current_model
    _current_model = _load_model(name_or_path)


def load_model_ephemeral(name_or_path: str) -> Dict:
    """Load a model by name or folder path without changing global current model."""
    return _load_model(name_or_path)


def get_current_model() -> Dict:
    global _current_model
    if _current_model is None:
        return {"type": "none", "name": "No Model Selected", "error": "Please select a model first"}
    return _current_model


def get_model_info() -> Dict:
    m = get_current_model()
    info = {"type": m.get("type"), "name": m.get("name", "Unknown Model")}
    if "path" in m:
        info["path"] = m["path"]
    if "fallback_error" in m:
        info["fallback_error"] = m["fallback_error"]
    if "error" in m:
        info["error"] = m["error"]
    return info