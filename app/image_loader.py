from pathlib import Path
import cv2
import numpy as np


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
    return cv2.equalizeHist(gray)


def clahe_contrast(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
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

    # Desaturate (blend grayscale)
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

    # CLAHE
    if params.get("clahe"):
        # Apply CLAHE on L channel of LAB for better color preservation
        lab = cv2.cvtColor(res, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        res = cv2.cvtColor(lab2, cv2.COLOR_Lab2BGR)

    # Equalize (on gray)
    if params.get("equalize"):
        eq = equalize_hist(res)
        res = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # Gradient overlay
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

    # Invert colors
    inv = params.get("invert", False)
    if isinstance(inv, str):
        inv = inv.lower() in ("1", "true", "yes", "on")
    if inv:
        res = cv2.bitwise_not(res)

    # Ensure uint8
    res = np.clip(res, 0, 255).astype(np.uint8)
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