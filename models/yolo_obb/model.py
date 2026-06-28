from typing import List, Dict, Any
import os
import yaml
import numpy as np
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None

# Optional: Load config if available
def load_config(model_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def load(config_override: Dict[str, Any] = None) -> Any:
    """
    Initialize and return the model object.
    """
    if YOLO is None:
        return {"error": "ultralytics not installed"}
        
    model_dir = os.path.dirname(__file__)
    config = load_config(model_dir)
    
    # Apply overrides
    if config_override:
        config.update(config_override)
    
    weights_file = config.get("weights_file", "yolov8n-obb.pt")
    # If not absolute, assume relative to model dir
    if not os.path.isabs(weights_file) and not weights_file.endswith(".pt"):
        # Could be a name like 'yolov8n-obb', let ultralytics handle it or append .pt
        pass
    elif not os.path.isabs(weights_file):
        weights_path = os.path.join(model_dir, weights_file)
        if os.path.exists(weights_path):
            weights_file = weights_path
            
    print(f"Loading YOLO-OBB from {weights_file}")
    try:
        model = YOLO(weights_file)
        
        # Determine device
        device = config.get("device", "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        
        # We don't strictly need to move it here, YOLO handles it in predict(), 
        # but we can store the device preference.
        return {"model": model, "config": config, "device": device}
    except Exception as e:
        return {"error": str(e)}

def infer(wrapper: Any, img: Any) -> List[Dict[str, float]]:
    """
    Run inference on an image (numpy array BGR).
    Returns a list of detections: [{"x": float, "y": float, "w": float, "h": float, "angle": float}, ...]
    """
    if "error" in wrapper:
        print(f"Model in error state: {wrapper['error']}")
        return []
        
    model = wrapper["model"]
    config = wrapper["config"]
    device = wrapper["device"]
    
    conf = config.get("confidence_threshold", 0.25)
    iou = config.get("iou_threshold", 0.45)
    imgsz = config.get("imgsz", 1024)
    max_det = config.get("max_det", 10000)
    
    # Run inference
    # verbose=False to keep stdout clean
    results = model.predict(
        img, device=device, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False
    )
    
    dets = []
    for r in results:
        # Check for OBB
        if hasattr(r, "obb") and r.obb is not None:
            # Move to CPU numpy
            xywhr = r.obb.xywhr.cpu().numpy() # [cx, cy, w, h, rotation_radians]
            conf_scores = r.obb.conf.cpu().numpy().ravel()
            
            for i in range(len(xywhr)):
                cx, cy, w, h, angle_rad = xywhr[i]
                dets.append({
                    "x": float(cx),
                    "y": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "angle": float(angle_rad),
                    "confidence": float(conf_scores[i])
                })
        # Fallback to boxes if no OBB
        elif hasattr(r, "boxes") and r.boxes is not None:
             boxes = r.boxes.xyxy.cpu().numpy()
             conf_scores = r.boxes.conf.cpu().numpy()
             for i, box in enumerate(boxes):
                 x1, y1, x2, y2 = box
                 cx = (x1 + x2) / 2.0
                 cy = (y1 + y2) / 2.0
                 w = x2 - x1
                 h = y2 - y1
                 dets.append({
                    "x": float(cx),
                    "y": float(cy),
                    "w": float(w),
                    "h": float(h),
                    "angle": 0.0,
                    "confidence": float(conf_scores[i])
                 })
                 
    return dets
