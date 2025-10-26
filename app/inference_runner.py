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
        # Pass tuned parameters to reduce NMS overhead and speed up inference
        try:
            results = yolo(img, verbose=False, conf=0.35, iou=0.5, max_det=150)
        except TypeError:
            # Fallback for older ultralytics versions not supporting kwargs
            results = yolo(img, verbose=False)
        dets: List[Dict] = []
        for r in results:
            # Check for oriented bounding boxes first (YOLO-OBB)
            if hasattr(r, "obb") and r.obb is not None:
                try:
                    # Convert tensor to numpy if needed
                    def _to_np(x):
                        try:
                            import torch
                            if isinstance(x, torch.Tensor):
                                return x.detach().cpu().numpy()
                        except Exception:
                            pass
                        return np.asarray(x)
                    
                    xywhr = _to_np(r.obb.xywhr)  # [cx, cy, w, h, rotation_radians]
                    conf = _to_np(r.obb.conf).ravel()
                    
                    for i in range(xywhr.shape[0]):
                        cx, cy, w, h, angle_rad = xywhr[i]
                        confidence = conf[i]
                        
                        # Keep center coordinates for proper rotated box drawing
                        dets.append({
                            "x": float(cx), 
                            "y": float(cy), 
                            "w": float(w), 
                            "h": float(h), 
                            "angle": float(angle_rad),  # Keep rotation angle in radians
                            "confidence": float(confidence)
                        })
                except Exception as e:
                    print(f"Error processing OBB: {e}")
                    
            # Fallback to regular bounding boxes if no OBB
            elif hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    dets.append({"x": cx, "y": cy, "w": w, "h": h, "angle": 0.0})
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
        
        # Convert angle from radians to degrees for OpenCV
        # YOLO-OBB returns angles in radians, but cv2.boxPoints expects degrees
        angle_deg = np.degrees(angle) if angle != 0.0 else 0.0
        
        rect = ((cx, cy), (max(w, 1.0), max(h, 1.0)), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
    return out