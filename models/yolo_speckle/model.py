"""
YOLO-OBB Speckle Detection Plugin for Crystal Analysis GUI

This plugin integrates the trained YOLO-OBB model for speckle detection.
It loads the model weights and performs inference on images, returning
detections in the format expected by the GUI system.
"""

import numpy as np
from pathlib import Path

# Model path - using the best weights from your trained model
MODEL_PATH = "/home/caiosiq/chem-gui/yolo_inference/runs/obb/gui_speckle_train/weights/best.pt"


def load():
    """
    Load the YOLO-OBB model.
    Returns the loaded model object.
    """
    try:
        from ultralytics import YOLO
        
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"YOLO model weights not found at {MODEL_PATH}")
        
        model = YOLO(MODEL_PATH)
        print(f"Loaded YOLO-OBB model from {MODEL_PATH}")
        return model
        
    except ImportError:
        raise ImportError("ultralytics package is required. Install with: pip install ultralytics")


def _to_np(x):
    """Convert tensor to numpy array if needed."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _extract_detections(results_obj):
    """
    Extract detections from YOLO results and convert to GUI format.
    Returns list of detection dictionaries with keys: x, y, w, h, angle
    Note: x,y must be CENTER coordinates (cx, cy) to match GUI drawing logic.
    """
    if not hasattr(results_obj, 'obb') or results_obj.obb is None:
        return []
    
    # Get oriented bounding boxes: [cx, cy, w, h, rotation_radians]
    xywhr = _to_np(results_obj.obb.xywhr)
    conf = _to_np(results_obj.obb.conf).ravel()
    cls = _to_np(results_obj.obb.cls).ravel()
    
    if xywhr.size == 0:
        return []
    
    detections = []
    for i in range(xywhr.shape[0]):
        cx, cy, w, h, angle_rad = xywhr[i]
        confidence = conf[i]
        class_id = cls[i]
        
        # Use center coordinates directly; GUI expects centers for rotated rectangles
        x = float(cx)
        y = float(cy)
        
        detection = {
            "x": x,
            "y": y, 
            "w": float(w),
            "h": float(h),
            "angle": float(angle_rad),  # radians as expected by GUI
            "confidence": float(confidence),
            "class": int(class_id)
        }
        detections.append(detection)
    
    return detections


def infer(model, img):
    """
    Run inference on an image using the YOLO-OBB model.
    
    Args:
        model: The loaded YOLO model from load()
        img: Input image as numpy array (BGR format)
    
    Returns:
        List of detection dictionaries with keys: x, y, w, h, angle
    """
    try:
        # Run YOLO inference
        results = model.predict(
            img,
            imgsz=1024,      # Use same size as training
            conf=0.30,       # Confidence threshold
            iou=0.30,        # IoU threshold for NMS
            verbose=False,
            save=False,
            max_det=10000
        )
        
        # Extract detections from first (and only) result
        if len(results) > 0:
            detections = _extract_detections(results[0])
            print(f"YOLO detected {len(detections)} objects")
            return detections
        else:
            return []
            
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return []