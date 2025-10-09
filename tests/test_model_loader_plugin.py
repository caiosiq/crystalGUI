from pathlib import Path
import numpy as np
import cv2

from app import model_loader
from app.inference_runner import run


def test_plugin_load_and_infer_blob():
    # Load example plugin
    root = Path(__file__).resolve().parents[1]
    plugin_dir = root / "models" / "example_blob"
    assert plugin_dir.exists(), "example_blob plugin folder missing"
    model_loader.set_current_model(str(plugin_dir))
    m = model_loader.get_current_model()
    assert m.get("type") == "plugin"

    # Create synthetic image with a white circle on black
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)

    dets = run(m, img)
    assert isinstance(dets, list)
    # Should find at least one detection near our circle center
    assert len(dets) >= 1
    cx = dets[0]["x"]
    cy = dets[0]["y"]
    assert 80 <= cx <= 120
    assert 80 <= cy <= 120