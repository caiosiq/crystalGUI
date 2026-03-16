from pathlib import Path
import os

def convert_dota_to_yolo(dataset_path: str, img_width: int, img_height: int):
    """
    Converts DOTA labels to YOLO OBB format.
    dataset_path: Path to the dataset root (containing 'labels_dota' folder).
    img_width: Image width for normalization.
    img_height: Image height for normalization.
    """
    root = Path(dataset_path)
    dota_dir = root / "labels_dota"
    out_dir = root / "labels"
    
    if not dota_dir.exists():
        raise FileNotFoundError(f"DOTA labels directory not found at {dota_dir}")
        
    out_dir.mkdir(exist_ok=True, parents=True)
    
    converted_count = 0
    
    for p in sorted(dota_dir.glob("*.txt")):
        try:
            lines = p.read_text().strip().splitlines()
            out_lines = []
            for ln in lines:
                if ln.startswith("imagesource") or ln.startswith("gsd"):
                    continue
                toks = ln.split()
                if len(toks) < 10:
                    continue
                
                # first 8 are x1 y1 ... x4 y4 (absolute pixels)
                xy = list(map(float, toks[:8]))
                
                # normalize
                x1, y1, x2, y2, x3, y3, x4, y4 = xy
                nx = [x1/img_width, x2/img_width, x3/img_width, x4/img_width]
                ny = [y1/img_height, y2/img_height, y3/img_height, y4/img_height]
                
                # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                out = [0, nx[0], ny[0], nx[1], ny[1], nx[2], ny[2], nx[3], ny[3]]  # class_id=0
                out_lines.append(" ".join(f"{v:.6f}" for v in out))
                
            (out_dir / p.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            converted_count += 1
        except Exception as e:
            print(f"Error converting {p.name}: {e}")
            
    return converted_count
