from __future__ import annotations

"""
CLI module to generate a batch of synthetic images using crystalGUI.osog.
Intended to be called via Slurm or a local background process by the GUI.
Generates images AND labels (DOTA and YOLO-OBB).
"""

import argparse
from pathlib import Path
import json
import cv2
import random
from crystalGUI.osog import generate_image, SynthConfig, sample_lambda, lambda_to_t
from crystalGUI.osog.utils.io import save_dota_label, save_yolo_obb

CLASS_NAME = "Crystal"
IMG_EXT = ".jpg"  # keep jpg for speed; labels are extension-agnostic


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
    cfg = SynthConfig.from_dict(cfg_dict)

    rng = random.Random(args.seed if args.seed is not None else (args.seed_base + args.index_offset))

    # Log the shard range for visibility
    print(f"[batch_job] index_offset={args.index_offset} n_images={args.n_images} -> indices [{args.index_offset} .. {args.index_offset + args.n_images - 1}]")

    generated = 0
    print(f"Starting loop for {args.n_images} images...", flush=True)
    for i in range(args.n_images):
        try:
            # Stage lambda sampling (log-uniform), then map to t
            lmbda = sample_lambda(rng, cfg)
            t = lambda_to_t(lmbda)
            
            # Deterministic per-image seed using seed_base + index_offset + i
            # Use global index for naming to avoid shard overwrites
            global_idx = args.index_offset + i
            per_seed = args.seed_base + global_idx
            
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
            
            if i % 10 == 0:
                print(f"Generated {i+1}/{args.n_images}", flush=True)
                
        except Exception as e:
            print(f"Error generating image {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"Generated {generated} new images into {images_dir} (labels written for all {args.n_images} indices in this shard)", flush=True)


if __name__ == "__main__":
    main()
