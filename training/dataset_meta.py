"""Dataset metadata for YOLO training (max_det from synth config + label scans)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from crystalGUI.osog.config import SynthConfig
from crystalGUI.osog.physics.stage import apply_stage_to_config, ensure_config

_META_FILENAME = "dataset_meta.json"
_CONFIG_FILENAME = "config.json"
_LABEL_DIRS = ("labels", "labels_yolo_obb")
_TIMESTAMP_RE = re.compile(r"_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$")


def particles_at_t1(cfg: Union[SynthConfig, dict]) -> int:
    """Labeled main-particle count at mature stage (t=1), before ghosts/fusion."""
    cfg = apply_stage_to_config(ensure_config(cfg), 1.0)
    phys = cfg.physics
    total = 0
    for attr in (
        "rod_specs",
        "sphere_specs",
        "cube_specs",
        "plate_specs",
        "polyhedra_specs",
        "bubble_specs",
        "droplet_specs",
    ):
        spec = getattr(phys, attr, None)
        if not spec or not getattr(spec, "enable", False):
            continue
        cr = spec.count_range
        if isinstance(cr, (list, tuple)) and len(cr) >= 2:
            total += int(max(cr[0], cr[1]))
        else:
            total += int(cr)
    return total


def recommended_max_det_from_config(cfg: Union[SynthConfig, dict]) -> int:
    """2× mature particle count — headroom for dense frames and minor fusion."""
    return max(1, 2 * particles_at_t1(cfg))


def scan_max_boxes_per_image(dataset_path: Union[str, Path]) -> int:
    """Maximum non-empty label lines in any YOLO label file under the dataset."""
    root = Path(dataset_path)
    max_boxes = 0
    for labels_name in _LABEL_DIRS:
        labels_root = root / labels_name
        if not labels_root.is_dir():
            continue
        for label_file in labels_root.rglob("*.txt"):
            if not label_file.is_file():
                continue
            try:
                n = sum(1 for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                continue
            max_boxes = max(max_boxes, n)
    return max_boxes


def _synth_jobs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "synth_jobs"


def _find_synth_config_for_dataset(dataset_path: Path) -> Optional[Path]:
    m = _TIMESTAMP_RE.search(dataset_path.name)
    if not m:
        return None
    ts = m.group(1)
    jobs_dir = _synth_jobs_dir()
    if not jobs_dir.is_dir():
        return None
    matches = sorted(jobs_dir.glob(f"{ts}_*/{_CONFIG_FILENAME}"))
    if matches:
        return matches[-1]
    return None


def load_synth_config(dataset_path: Union[str, Path]) -> Optional[dict]:
    root = Path(dataset_path)
    for candidate in (root / _CONFIG_FILENAME, _find_synth_config_for_dataset(root)):
        if candidate and candidate.is_file():
            try:
                with candidate.open(encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
    return None


def read_dataset_meta(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    meta_path = Path(dataset_path) / _META_FILENAME
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_dataset_meta(dataset_path: Union[str, Path], cfg: Optional[Union[SynthConfig, dict]] = None) -> Dict[str, Any]:
    """Write or update dataset_meta.json from synth config."""
    root = Path(dataset_path)
    root.mkdir(parents=True, exist_ok=True)
    meta = read_dataset_meta(root)
    if cfg is None:
        cfg = load_synth_config(root)
    if cfg is not None:
        n_t1 = particles_at_t1(cfg)
        meta["n_particles_at_t1"] = n_t1
        meta["recommended_max_det"] = recommended_max_det_from_config(cfg)
        fused = (cfg.get("physics") or {}).get("fused") or {}
        if fused.get("enable"):
            meta["note"] = (
                "Agglomeration can produce more labeled boxes than n_particles_at_t1; "
                "training_max_det also considers max_boxes_per_image."
            )
    meta_path = root / _META_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def refresh_observed_max_boxes(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """Scan labels and update max_boxes_per_image / training_max_det in meta."""
    root = Path(dataset_path)
    meta = read_dataset_meta(root)
    if not meta.get("recommended_max_det"):
        write_dataset_meta(root)
        meta = read_dataset_meta(root)
    observed = scan_max_boxes_per_image(root)
    meta["max_boxes_per_image"] = observed
    meta["training_max_det"] = resolve_training_max_det(root, meta=meta)
    meta_path = root / _META_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def resolve_training_max_det(
    dataset_path: Union[str, Path],
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Training max_det: max(2×N@t=1, observed boxes in labels, 300 Ultralytics default).
    """
    root = Path(dataset_path)
    if meta is None:
        meta = read_dataset_meta(root)
    recommended = int(meta.get("recommended_max_det") or 0)
    if recommended <= 0:
        cfg = load_synth_config(root)
        if cfg is not None:
            recommended = recommended_max_det_from_config(cfg)
    observed = int(meta.get("max_boxes_per_image") or 0)
    if observed <= 0:
        observed = scan_max_boxes_per_image(root)
    return max(300, recommended, observed)
