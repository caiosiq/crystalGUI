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