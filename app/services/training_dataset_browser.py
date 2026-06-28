"""Paginated training dataset image listing and label parsing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_DATASET_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_SPLIT_NAMES = ("train", "val", "test")

# (dataset_key, split) -> (images_dir_mtime, [(stem, rel_image_path), ...])
_image_index_cache: dict[tuple[str, str], tuple[float, list[tuple[str, str]]]] = {}


def _is_split_dataset(root: Path) -> bool:
    return (root / "images" / "train").is_dir()


def _images_dir(root: Path, split: str) -> Path | None:
    if _is_split_dataset(root):
        if split not in _SPLIT_NAMES:
            return None
        return root / "images" / split
    if split != "all":
        return None
    return root / "images"


def split_counts(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if _is_split_dataset(root):
        for name in _SPLIT_NAMES:
            counts[name] = _count_images_in_dir(root / "images" / name)
    else:
        counts["all"] = _count_images_in_dir(root / "images")
    return counts


def _count_images_in_dir(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        1 for fp in path.iterdir()
        if fp.is_file() and fp.suffix.lower() in _DATASET_IMAGE_EXTS
    )


def _build_image_index(root: Path, split: str) -> list[tuple[str, str]]:
    img_dir = _images_dir(root, split)
    if img_dir is None or not img_dir.is_dir():
        return []

    cache_key = (str(root.resolve()), split)
    mtime = os.path.getmtime(img_dir)
    cached = _image_index_cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    items: list[tuple[str, str]] = []
    for fp in sorted(img_dir.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in _DATASET_IMAGE_EXTS:
            continue
        rel = fp.relative_to(root).as_posix()
        items.append((fp.stem, rel))

    _image_index_cache[cache_key] = (mtime, items)
    return items


def resolve_dataset_file(root: Path, rel_path: str) -> Path:
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid dataset file path") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"File not found: {rel_path}")
    return candidate


def _label_candidates(root: Path, split: str, stem: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    if _is_split_dataset(root):
        yolo = root / "labels" / split / f"{stem}.txt"
        if yolo.is_file():
            paths.append(("yolo", yolo))
    else:
        yolo = root / "labels" / f"{stem}.txt"
        if yolo.is_file():
            paths.append(("yolo", yolo))

    dota = root / "labels_dota" / f"{stem}.txt"
    if dota.is_file():
        paths.append(("dota", dota))
    return paths


def parse_label_file(fmt: str, text: str, img_w: int, img_h: int) -> list[dict[str, Any]]:
    obbs: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        if fmt == "dota":
            if line.startswith("imagesource") or line.startswith("gsd"):
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            xy = list(map(float, toks[:8]))
            corners = [
                [xy[0], xy[1]], [xy[2], xy[3]],
                [xy[4], xy[5]], [xy[6], xy[7]],
            ]
            obbs.append({"corners": corners})
        else:
            toks = line.split()
            if len(toks) < 9:
                continue
            coords = list(map(float, toks[1:9]))
            corners = [
                [coords[i] * img_w, coords[i + 1] * img_h]
                for i in range(0, 8, 2)
            ]
            obbs.append({"corners": corners})
    return obbs


def load_obbs_for_image(root: Path, split: str, stem: str, img_w: int, img_h: int) -> list[dict[str, Any]]:
    for fmt, path in _label_candidates(root, split, stem):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        obbs = parse_label_file(fmt, text, img_w, img_h)
        if obbs:
            return obbs
    return []


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return 1024, 1024


def list_samples(
    root: Path,
    split: str,
    offset: int,
    limit: int,
    *,
    include_labels: bool = False,
) -> dict[str, Any]:
    is_split = _is_split_dataset(root)
    effective_split = split if is_split else "all"
    if is_split and split not in _SPLIT_NAMES:
        raise ValueError(f"Invalid split '{split}' for a split dataset")
    if not is_split and split != "all":
        effective_split = "all"

    index = _build_image_index(root, effective_split)
    total = len(index)
    offset = max(0, min(offset, max(0, total - 1))) if total else 0
    limit = max(1, min(limit, 50))
    slice_items = index[offset: offset + limit]

    items = []
    for i, (stem, rel_image) in enumerate(slice_items):
        entry: dict[str, Any] = {
            "index": offset + i,
            "stem": stem,
            "image_rel": rel_image,
        }
        if include_labels:
            img_path = root / rel_image
            w, h = get_image_dimensions(img_path)
            entry["width"] = w
            entry["height"] = h
            entry["obbs"] = load_obbs_for_image(root, effective_split, stem, w, h)
        items.append(entry)

    return {
        "split": effective_split,
        "is_split": is_split,
        "total": total,
        "offset": offset,
        "limit": limit,
        "split_counts": split_counts(root),
        "items": items,
    }
