from pathlib import Path
from typing import List, Dict, Any
import math

CLASS_NAME = "Crystal"

def save_dota_label(txt_path: Path, obbs: List[Dict[str, Any]]) -> None:
    """Write DOTA-style quadrilateral labels using recorded corners."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("imagesource:GoogleEarth\n")
        f.write("gsd:1\n")
        for ob in obbs:
            corners = ob.get("corners")
            if not corners or len(corners) != 4:
                continue
            coords = " ".join([f"{float(x):.2f} {float(y):.2f}" for x, y in corners])
            f.write(f"{coords} {CLASS_NAME} 0\n")

def save_yolo_obb(txt_path: Path, obbs: List[Dict[str, Any]], img_w: int, img_h: int) -> None:
    """Write YOLO-OBB labels: class_id cx cy w h angle(rad), normalized to image size."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for ob in obbs:
            cx = float(ob.get("cx", 0.0))
            cy = float(ob.get("cy", 0.0))
            L = float(ob.get("L", 0.0))  # major axis length
            W = float(ob.get("W", 0.0))  # minor axis length
            ang_deg = float(ob.get("angle_deg", 0.0))
            # normalize
            cx_n = cx / max(1e-6, img_w)
            cy_n = cy / max(1e-6, img_h)
            w_n = L / max(1e-6, img_w)
            h_n = W / max(1e-6, img_h)
            ang_rad = math.radians(ang_deg)
            f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} {ang_rad:.6f}\n")
