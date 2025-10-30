from pathlib import Path
import cv2
import numpy as np
import os
import time

# Enable OpenCV optimizations, OpenCL (if available), and multi-threading
cv2.setUseOptimized(True)
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass
try:
    _threads = int(os.getenv("OPENCV_THREADS", "0")) or (os.cpu_count() or 0)
    if _threads and _threads > 0:
        cv2.setNumThreads(_threads)
except Exception:
    pass

# Tunables via environment
def _get_clahe_tunables():
    try:
        clip = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
    except Exception:
        clip = 2.0
    try:
        grid = int(os.getenv("CLAHE_TILE_GRID", "8"))
        grid = max(1, min(grid, 64))
    except Exception:
        grid = 8
    return clip, grid


def _cuda_available() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _upload_to_gpu(arr: np.ndarray):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(arr)
    return gpu


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_hist(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    # Try GPU acceleration if available
    if _cuda_available() and hasattr(cv2.cuda, "equalizeHist"):
        try:
            gpu = _upload_to_gpu(gray)
            gpu_eq = cv2.cuda.equalizeHist(gpu)
            return gpu_eq.download()
        except Exception:
            # Fallback to CPU
            pass
    return cv2.equalizeHist(gray)


def clahe_contrast(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    clip, grid = _get_clahe_tunables()
    # Try GPU acceleration if available
    if _cuda_available() and hasattr(cv2.cuda, "createCLAHE"):
        try:
            clahe = cv2.cuda.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            gpu = _upload_to_gpu(gray)
            gpu_res = clahe.apply(gpu)
            return gpu_res.download()
        except Exception:
            # Fallback to CPU
            pass
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    # Prefer 16-bit intermediate to reduce CPU cost compared to 64F
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    # Convert to float32 for magnitude computation
    mag = np.sqrt(np.float32(gx) * np.float32(gx) + np.float32(gy) * np.float32(gy))
    mag = np.uint8(255 * (mag / (mag.max() + 1e-6)))
    return mag


def apply_operation(img: np.ndarray, operation: str) -> np.ndarray:
    op = operation.lower()
    if op in ["gray", "grayscale"]:
        return to_grayscale(img)
    if op in ["equalize", "hist_eq", "histogram_equalization"]:
        return equalize_hist(img)
    if op in ["contrast", "clahe"]:
        return clahe_contrast(img)
    if op in ["blur", "gaussian"]:
        return gaussian_blur(img)
    if op in ["denoise", "nlmeans"]:
        return denoise(img)
    if op in ["gradient", "sobel"]:
        return gradient_magnitude(img)
    # default: return original
    return img


def load_image_bytes(buf: bytes) -> np.ndarray:
    """Decode image from bytes into a BGR cv2 image."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes")
    return img


def ensure_color(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR. If grayscale, convert to BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    # Fallback: attempt to convert single channel
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def apply_pipeline(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply a parameterized preprocessing pipeline.
    Supported params:
      - desaturate: float [0,1] amount of grayscale blending
      - invert: bool, invert colors
      - gradient_strength: float [0,1], overlay gradient magnitude
      - clahe: bool, apply CLAHE (acts on L channel or gray)
      - equalize: bool, histogram equalization (gray)
    Returns a 3-channel BGR image suitable for model inference.
    """
    if params is None:
        params = {}
    res = ensure_color(img.copy())

    timings_on = os.getenv("PREPROC_TIMING", "0") == "1"
    timings = {}

    # Desaturate (blend grayscale)
    t0 = time.perf_counter()
    desat = params.get("desaturate", 0.0)
    try:
        desat = float(desat)
    except Exception:
        desat = 0.0
    # Allow 0-100 inputs
    if desat > 1.0:
        desat = min(desat / 100.0, 1.0)
    if desat > 0:
        gray = to_grayscale(res)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        res = cv2.addWeighted(res, 1.0 - desat, gray3, desat, 0)
    if timings_on:
        timings["desaturate"] = time.perf_counter() - t0

    # CLAHE
    t1 = time.perf_counter()
    if params.get("clahe"):
        # Apply CLAHE on L channel of LAB for better color preservation
        lab = cv2.cvtColor(res, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l2 = None
        # Get defaults from environment, allow override via params
        clip, grid = _get_clahe_tunables()
        try:
            if "clahe_clip_limit" in params:
                clip = float(params.get("clahe_clip_limit", clip))
            if "clahe_tile_grid" in params:
                grid = int(params.get("clahe_tile_grid", grid))
                grid = max(1, min(grid, 64))
        except Exception:
            # If parsing fails, keep environment defaults
            pass
        if _cuda_available() and hasattr(cv2.cuda, "createCLAHE"):
            try:
                clahe = cv2.cuda.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
                gpu_l = _upload_to_gpu(l)
                gpu_l2 = clahe.apply(gpu_l)
                l2 = gpu_l2.download()
            except Exception:
                l2 = None
        if l2 is None:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        res = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)
    if timings_on:
        timings["clahe"] = time.perf_counter() - t1

    # Equalize (on gray)
    t2 = time.perf_counter()
    if params.get("equalize"):
        eq = equalize_hist(res)
        res = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if timings_on:
        timings["equalize"] = time.perf_counter() - t2

    # Gradient overlay
    t3 = time.perf_counter()
    grad_str = params.get("gradient_strength", 0.0)
    try:
        grad_str = float(grad_str)
    except Exception:
        grad_str = 0.0
    if grad_str > 1.0:
        grad_str = min(grad_str / 100.0, 1.0)
    if grad_str > 0:
        grad = gradient_magnitude(res)
        # Slight blur to avoid harsh edges
        grad = cv2.GaussianBlur(grad, (3, 3), 0)
        grad3 = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        # Overlay edges by addition-weighted (brighten edges)
        res = cv2.addWeighted(res, 1.0, grad3, grad_str, 0)
    if timings_on:
        timings["gradient_overlay"] = time.perf_counter() - t3

    # Invert colors
    t4 = time.perf_counter()
    inv = params.get("invert", False)
    if isinstance(inv, str):
        inv = inv.lower() in ("1", "true", "yes", "on")
    if inv:
        res = cv2.bitwise_not(res)
    if timings_on:
        timings["invert"] = time.perf_counter() - t4

    # Ensure uint8
    res = np.clip(res, 0, 255).astype(np.uint8)

    if timings_on:
        total = sum(timings.values())
        h, w = res.shape[:2]
        print(f"[PREPROC_TIMING] desat={timings.get('desaturate', 0):.3f}s, clahe={timings.get('clahe', 0):.3f}s, "
              f"equalize={timings.get('equalize', 0):.3f}s, grad={timings.get('gradient_overlay', 0):.3f}s, "
              f"invert={timings.get('invert', 0):.3f}s, total={total:.3f}s, size=({w}x{h}), threads={cv2.getNumThreads()}")

    return res


# (Deduplicated) encode_image_to_base64 is defined above.


def encode_image_to_base64(img: np.ndarray, fmt: str = "jpeg", quality: int = 80) -> str:
    """
    Encode np.ndarray (BGR) image to base64 data URL using OpenCV.
    fmt: 'png' or 'jpeg'/'jpg'. Default JPEG for smaller payloads.
    quality: JPEG quality 0-100.
    """
    import base64
    # Ensure proper color shape
    enc_img = ensure_color(img)
    fmt = (fmt or "jpeg").lower()
    if fmt in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(quality, 100)))]
        ok, buf = cv2.imencode('.jpg', enc_img, params)
        mime = 'image/jpeg'
    else:
        ok, buf = cv2.imencode('.png', enc_img)
        mime = 'image/png'
    if not ok:
        raise ValueError("Failed to encode image to base64")
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:{mime};base64,{b64}"