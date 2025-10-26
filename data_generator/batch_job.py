from __future__ import annotations

"""
CLI module to generate a batch of synthetic images using crystalGUI.data_generator.synth.
Intended to be called via Slurm or a local background process by the GUI.
Generates images AND labels (DOTA and YOLO-OBB) similar to synth_speckles/batched_full_synth.py.
"""

import argparse
from pathlib import Path
import json
import cv2
import math
import random
from typing import List, Dict, Any
from .synth import generate_image, SynthConfig, sample_lambda, lambda_to_t

CLASS_NAME = "Crystal"
IMG_EXT = ".jpg"  # keep jpg for speed; labels are extension-agnostic


def save_dota_label(txt_path: Path, obbs: List[Dict[str, Any]]) -> None:
    """Write DOTA-style quadrilateral labels using recorded corners from obb_list."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--config-file", type=str, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=0, help="Base seed for deterministic per-image seeding (parallelization)")
    ap.add_argument("--index-offset", type=int, default=0, help="Index offset for this shard")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    # Subfolders to mirror batched_full_synth.py
    images_dir = out_dir / "images"
    dota_dir = out_dir / "labels_dota"
    yolo_dir = out_dir / "labels_yolo_obb"
    images_dir.mkdir(parents=True, exist_ok=True)
    dota_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # Write classes.txt (same as the speckles generator)
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write(f"{CLASS_NAME}\n")

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg = SynthConfig(**cfg_dict)

    rng = random.Random(args.seed if args.seed is not None else (args.seed_base + args.index_offset))

    # Log the shard range for visibility
    print(f"[batch_job] index_offset={args.index_offset} n_images={args.n_images} -> indices [{args.index_offset} .. {args.index_offset + args.n_images - 1}]")

    generated = 0
    for i in range(args.n_images):
        # Stage lambda sampling (log-uniform), then map to t
        lmbda = sample_lambda(rng, cfg)
        t = lambda_to_t(lmbda)
        # Deterministic per-image seed using seed_base + index_offset + i
        per_seed = (args.seed_base + args.index_offset + i) if (args.seed_base or args.index_offset) else None
        # Use global index for naming to avoid shard overwrites
        global_idx = args.index_offset + i
        stem = f"{global_idx:08d}"
        img_path = images_dir / f"{stem}{IMG_EXT}"
        dota_path = dota_dir / f"{stem}.txt"
        yolo_path = yolo_dir / f"{stem}.txt"

        # Always compute image + OBBs deterministically (so we can write labels even if image exists)
        img, obbs = generate_image(cfg_dict, t, seed=per_seed, return_obbs=True)

        # Write image if not present
        if not img_path.exists():
            cv2.imwrite(str(img_path), img)
            generated += 1
        # Write labels (DOTA + YOLO-OBB)
        save_dota_label(dota_path, obbs)
        save_yolo_obb(yolo_path, obbs, img_w=cfg.width, img_h=cfg.height)

    print(f"Generated {generated} new images into {images_dir} (labels written for all {args.n_images} indices in this shard)")


if __name__ == "__main__":
    main()