from pathlib import Path
import json
import yaml
import os
import shutil
import uuid
import datetime

from crystalGUI.training import dataset_meta as training_dataset_meta

def generate_dataset_yaml(dataset_path: str, class_names: dict = {0: "Crystal"}):
    """
    Generates a YOLO dataset.yaml file inside the dataset directory.
    """
    root = Path(dataset_path).resolve()
    yaml_path = root / "dataset.yaml"
    
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
        
    return str(yaml_path)

def generate_training_slurm(
    dataset_path: str,
    model_name: str = "yolo11n-obb.pt",
    epochs: int = 100,
    batch_size: int = 4,
    img_size: int = 1024,
    workers: int = 8,
    project_name: str = "gui_train",
    slurm_config: dict = None,
    max_det=None,
):
    """
    Generates a Slurm script for training YOLO OBB model.
    """
    if slurm_config is None:
        slurm_config = {}
        
    dataset_root = Path(dataset_path).resolve()
    # Generate YAML first
    data_yaml = generate_dataset_yaml(dataset_path)

    meta = training_dataset_meta.refresh_observed_max_boxes(dataset_root)
    if max_det is None:
        max_det = int(meta.get("training_max_det") or training_dataset_meta.resolve_training_max_det(dataset_root))
    
    # Define output directory for runs (relative to dataset or central runs dir?)
    # User's script used: /home/caiosiq/chem-gui/yolo_inference/runs/obb
    # Let's use a 'runs' folder inside the dataset for self-containment, 
    # OR keep a central runs folder if preferred.
    # Given the user wants to "save it somewhere (that the UI can find it)", 
    # a central `runs` folder in `crystalGUI/data/runs` might be better organized 
    # than scattering runs inside dataset folders.
    
    # Let's use `crystalGUI/data/runs` as the central location.
    # crystalGUI/training/slurm_utils.py -> crystalGUI/
    base_dir = Path(__file__).resolve().parent.parent
    runs_dir = base_dir / "data" / "runs" / "obb"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = base_dir / "data" / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    job_name = f"train_{Path(dataset_path).name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    slurm_path = logs_dir / f"{job_name}.slurm"
    
    # Slurm defaults
    partition = slurm_config.get("partition", "mit_preemptable")
    time_limit = slurm_config.get("time", "06:00:00")
    gpu = slurm_config.get("gpu", "h200:1") # Default from user example
    cpus = slurm_config.get("cpus", 8)
    mem = slurm_config.get("mem", "64G")
    
    script_content = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p {partition}
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH -o {logs_dir}/{job_name}_%j.out
#SBATCH -e {logs_dir}/{job_name}_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate base

# Go to yolo_inference directory or project root?
# User's script went to /home/caiosiq/chem-gui/yolo_inference
# We can just run from anywhere if environment is set up, 
# but let's cd to crystalGUI root to be safe if we need local modules.
cd {base_dir}

echo "Starting training job {job_name}"
echo "Dataset: {data_yaml}"
echo "Output: {runs_dir}/{project_name}"
echo "max_det: {max_det}"

yolo obb train \\
  model={model_name} \\
  data="{data_yaml}" \\
  imgsz={img_size} \\
  batch={batch_size} \\
  epochs={epochs} \\
  device=0 \\
  workers={workers} \\
  max_det={max_det} \\
  project="{runs_dir}" \\
  name="{project_name}" \\
  exist_ok=True

echo "Training finished"
"""

    with open(slurm_path, "w") as f:
        f.write(script_content)
        
    return str(slurm_path), str(runs_dir / project_name), job_name, max_det
