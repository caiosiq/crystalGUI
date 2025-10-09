from typing import List, Dict
import numpy as np


def compute_stats(dets: List[Dict]) -> Dict:
    count = len(dets)
    if count == 0:
        return {"count": 0, "mean_area": 0.0, "mean_aspect_ratio": 0.0, "areas": [], "aspect_ratios": []}

    areas = []
    ars = []
    for d in dets:
        w = max(float(d.get("w", 0.0)), 1e-6)
        h = max(float(d.get("h", 0.0)), 1e-6)
        areas.append(w * h)
        ars.append(w / h)

    areas_np = np.array(areas, dtype=np.float32)
    ars_np = np.array(ars, dtype=np.float32)
    return {
        "count": int(count),
        "mean_area": float(areas_np.mean()),
        "mean_aspect_ratio": float(ars_np.mean()),
        "areas": areas,
        "aspect_ratios": ars,
    }