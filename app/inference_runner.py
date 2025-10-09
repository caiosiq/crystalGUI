from typing import List, Dict
import cv2
import numpy as np


def run(model: Dict, img: np.ndarray) -> List[Dict]:
    mtype = model.get("type")
    if mtype == "yolo":
        # YOLO inference path (if ultralytics installed)
        yolo = model.get("model")
        if yolo is None:
            return []
        # results: list of ultralytics Results
        results = yolo(img, verbose=False)
        dets: List[Dict] = []
        for r in results:
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    w = x2 - x1
                    h = y2 - y1
                    dets.append({"x": x1, "y": y1, "w": w, "h": h, "angle": 0.0})
        return dets

    if mtype == "plugin":
        mod = model.get("module")
        mdl = model.get("model")
        if mod and hasattr(mod, "infer"):
            try:
                return mod.infer(mdl, img)
            except Exception:
                pass

    # Default: classical image processing to find crystals via contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu threshold + morphology to connect fragments
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[Dict] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        dets.append({"x": float(cx), "y": float(cy), "w": float(w), "h": float(h), "angle": float(angle)})
    return dets


def draw_detections(img: np.ndarray, dets: List[Dict]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        cx, cy, w, h, angle = d["x"], d["y"], d["w"], d["h"], d["angle"]
        rect = ((cx, cy), (max(w, 1.0), max(h, 1.0)), angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
    return out