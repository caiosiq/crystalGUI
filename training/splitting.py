from pathlib import Path
import shutil
import random
import os

def split_dataset(dataset_path: str, ratios: list = [0.8, 0.1, 0.1]):
    """
    Splits dataset into train/val/test subsets.
    dataset_path: Path to dataset root (containing 'images' and 'labels' folders).
    ratios: List of [train_ratio, val_ratio, test_ratio].
    """
    root = Path(dataset_path)
    images_dir = root / "images"
    labels_dir = root / "labels"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    # Supported extensions
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    imgs = []
    for p in sorted(images_dir.glob("*")):
        if p.suffix.lower() in exts and p.is_file():
            imgs.append(p)
            
    if not imgs:
        return {"train": 0, "val": 0, "test": 0, "total": 0}

    # Shuffle for random split
    # Use a fixed seed for reproducibility if needed, but for splitting usually random is fine.
    # We can pass a seed if we want strict determinism.
    random.seed(42) 
    random.shuffle(imgs)
    
    n = len(imgs)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    # Remaining go to test
    
    splits = [
        ("train", imgs[:n_tr]), 
        ("val", imgs[n_tr:n_tr+n_va]), 
        ("test", imgs[n_tr+n_va:])
    ]
    
    counts = {}
    
    for split_name, items in splits:
        split_img_dir = images_dir / split_name
        split_lbl_dir = labels_dir / split_name
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for ip in items:
            # Move image
            try:
                shutil.move(str(ip), str(split_img_dir / ip.name))
                
                # Move corresponding label if exists
                lp = labels_dir / (ip.stem + ".txt")
                if lp.exists():
                    shutil.move(str(lp), str(split_lbl_dir / lp.name))
                
                count += 1
            except Exception as e:
                print(f"Error moving {ip.name}: {e}")
                
        counts[split_name] = count
        
    counts["total"] = n
    return counts
