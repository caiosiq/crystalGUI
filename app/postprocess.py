from typing import List, Dict
import numpy as np


def compute_stats(dets: List[Dict]) -> Dict:
    count = len(dets)
    if count == 0:
        return {
            "count": 0,
            "mean_length": 0.0,
            "mean_width": 0.0,
            "mean_aspect_ratio": 0.0,
            "lengths": [],
            "widths": [],
            "aspect_ratios": [],
            # legacy fields for compatibility
            "mean_area": 0.0,
            "areas": [],
        }

    lengths = []
    widths = []
    ars = []
    areas = []
    for d in dets:
        # Ensure positive dimensions
        w = max(float(d.get("w", 0.0)), 1e-6)
        h = max(float(d.get("h", 0.0)), 1e-6)
        # Define length as the larger dimension, width as the smaller
        L = max(w, h)
        W = min(w, h)
        lengths.append(L)
        widths.append(W)
        ars.append(L / W)
        areas.append(w * h)  # legacy: keep area if needed elsewhere

    lengths_np = np.array(lengths, dtype=np.float32)
    widths_np = np.array(widths, dtype=np.float32)
    ars_np = np.array(ars, dtype=np.float32)
    areas_np = np.array(areas, dtype=np.float32)

    return {
        "count": int(count),
        "mean_length": float(lengths_np.mean()),
        "mean_width": float(widths_np.mean()),
        "mean_aspect_ratio": float(ars_np.mean()),
        "lengths": lengths,
        "widths": widths,
        "aspect_ratios": ars,
        # legacy summary for compatibility (not used by new UI)
        "mean_area": float(areas_np.mean()),
        "areas": areas,
    }