from pathlib import Path
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app, UPLOADS_DIR


client = TestClient(app)


def _write_sample_upload(name: str = "test_img.png") -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOADS_DIR / name
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(img, (32, 32), (96, 96), (255, 255, 255), -1)
    # Write using OpenCV
    cv2.imwrite(str(p), img)
    return name


def test_index_renders():
    res = client.get("/")
    assert res.status_code == 200
    assert "Crystal Analysis GUI" in res.text


def test_inference_on_uploaded_image():
    name = _write_sample_upload()
    form = {"image_name": name}
    res = client.post("/inference", data=form)
    data = res.json()
    assert res.status_code == 200
    assert data["ok"] is True
    assert "overlay_url" in data
    assert "stats" in data