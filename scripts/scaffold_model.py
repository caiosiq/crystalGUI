import argparse
import os
from pathlib import Path

MODEL_TEMPLATE = """from typing import List, Dict, Any
import os
import yaml

# Optional: Load config if available
def load_config(model_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def load() -> Any:
    \"\"\"
    Initialize and return the model object.
    This function is optional. If not provided, 'model' will be None in infer().
    \"\"\"
    # Example: Load weights, initialize device
    # config = load_config(os.path.dirname(__file__))
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # return MyModel(weights=..., device=device)
    print("Loading model...")
    return {"status": "loaded"}

def infer(model: Any, img: Any) -> List[Dict[str, float]]:
    \"\"\"
    Run inference on an image (numpy array BGR).
    Returns a list of detections: [{"x": float, "y": float, "w": float, "h": float, "angle": float}, ...]
    \"\"\"
    # Example:
    # results = model.predict(img)
    # return [{"x": 100.0, "y": 100.0, "w": 50.0, "h": 20.0, "angle": 0.0}]
    return []
"""

CONFIG_TEMPLATE = """# Model Configuration
device: "cpu" # or "cuda:0"
confidence_threshold: 0.5
iou_threshold: 0.45
"""

def main():
    parser = argparse.ArgumentParser(description="Scaffold a new OSOG model plugin.")
    parser.add_argument("name", help="Name of the model (folder name)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / args.name
    if model_path.exists():
        print(f"Error: Model folder '{args.name}' already exists in {models_dir}")
        return

    model_path.mkdir()
    
    # Write model.py
    with open(model_path / "model.py", "w") as f:
        f.write(MODEL_TEMPLATE)
    
    # Write config.yaml
    with open(model_path / "config.yaml", "w") as f:
        f.write(CONFIG_TEMPLATE)
        
    # Write name.txt
    display_name = args.name.replace("_", " ").title()
    with open(model_path / "name.txt", "w") as f:
        f.write(display_name)

    print(f"Successfully created model plugin '{args.name}' at {model_path}")
    print("Files created:")
    print(f"  - {model_path / 'model.py'}")
    print(f"  - {model_path / 'config.yaml'}")
    print(f"  - {model_path / 'name.txt'}")

if __name__ == "__main__":
    main()
