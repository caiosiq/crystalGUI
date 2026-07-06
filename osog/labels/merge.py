"""Merge overlapping OBB labels into simplified crystals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _corners_array(ob: Dict[str, Any]) -> Optional[np.ndarray]:
    corners = ob.get("corners")
    if not corners or len(corners) != 4:
        return None
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        return None
    return pts


def polygon_area(corners: np.ndarray) -> float:
    return float(cv2.contourArea(corners.reshape(-1, 1, 2).astype(np.float32)))


def overlap_fraction_smaller_box(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    """
    Intersection area divided by the smaller of the two box areas.
    Returns 0 when boxes do not overlap.
    """
    area_a = polygon_area(corners_a)
    area_b = polygon_area(corners_b)
    if area_a < 1e-6 or area_b < 1e-6:
        return 0.0
    ret, intersect = cv2.intersectConvexConvex(
        corners_a.astype(np.float32),
        corners_b.astype(np.float32),
    )
    if ret <= 0 or intersect is None:
        return 0.0
    inter_pts = np.asarray(intersect, dtype=np.float32).reshape(-1, 2)
    if inter_pts.shape[0] < 3:
        return 0.0
    inter_area = float(cv2.contourArea(inter_pts.reshape(-1, 1, 2)))
    if inter_area <= 0.0:
        return 0.0
    return inter_area / min(area_a, area_b)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _merge_cluster(members: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(members) == 1:
        return dict(members[0])

    all_pts: List[List[float]] = []
    for m in members:
        pts = _corners_array(m)
        if pts is not None:
            all_pts.extend(pts.tolist())

    if len(all_pts) < 3:
        return dict(members[0])

    arr = np.asarray(all_pts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)
    (cx, cy), (w, h), ang = rect
    L, W = max(w, h), min(w, h)
    corners = cv2.boxPoints(rect)

    rep = max(members, key=lambda m: float(m.get("L", 0.0)) * float(m.get("W", 0.0)))
    merged = dict(rep)
    merged.update(
        {
            "cx": float(cx),
            "cy": float(cy),
            "L": float(L),
            "W": float(W),
            "angle_deg": float(ang),
            "corners": corners.tolist(),
            "merge_count": len(members),
        }
    )
    return merged


def merge_obbs(
    obbs: List[Dict[str, Any]],
    overlap_threshold: float,
    *,
    merge_by_group_id: bool = False,
) -> List[Dict[str, Any]]:
    """
    Cluster OBBs with overlap_fraction >= threshold (intersection / min area).
    Each cluster becomes one min-area OBB covering all member corners.
    """
    if not obbs or overlap_threshold <= 0.0:
        return list(obbs)

    n = len(obbs)
    corners_list: List[Optional[np.ndarray]] = [_corners_array(ob) for ob in obbs]
    valid = [i for i, c in enumerate(corners_list) if c is not None]
    if len(valid) <= 1:
        return list(obbs)

    uf = _UnionFind(n)
    thr = float(max(0.0, min(1.0, overlap_threshold)))

    for ii in range(len(valid)):
        i = valid[ii]
        for jj in range(ii + 1, len(valid)):
            j = valid[jj]
            if merge_by_group_id:
                gi = obbs[i].get("group_id", -1)
                gj = obbs[j].get("group_id", -1)
                if gi != gj or gi is None or gi < 0:
                    continue
            frac = overlap_fraction_smaller_box(corners_list[i], corners_list[j])
            if frac >= thr:
                uf.union(i, j)

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(obbs[idx])

    merged = [_merge_cluster(members) for members in clusters.values()]
    merged.sort(key=lambda ob: (float(ob.get("cy", 0.0)), float(ob.get("cx", 0.0))))
    return merged
