
import unittest
import shutil
import tempfile
import cv2
import numpy as np
import time
import json
from pathlib import Path
from app.services.calibration_manager import CalibrationManager
from crystalGUI.osog.config import SynthConfig

class TestCalibrationManager(unittest.TestCase):
    def setUp(self):
        # Create temporary directory structure
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.results_dir = self.data_dir / "results"
        self.uploads_dir.mkdir(parents=True)
        self.results_dir.mkdir(parents=True)
        
        # Create a dummy target image
        self.target_img_path = self.uploads_dir / "target.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(self.target_img_path), img)
        
        # Initialize Manager
        self.manager = CalibrationManager(self.test_dir)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_start_job(self):
        # Config matching image size to avoid resize prints
        cfg = SynthConfig()
        cfg.canvas.width = 100
        cfg.canvas.height = 100
        
        job_id = self.manager.start_job(
            target_image_path=str(self.target_img_path),
            initial_config=cfg.to_dict(),
            selected_params=["optics.focus_z"],
            max_steps=5, # Short run
            learning_rate=0.01,
            device="cpu"
        )
        
        print(f"Started job: {job_id}")
        self.assertIsNotNone(job_id)
        
        # Poll status
        for _ in range(20):
            status = self.manager.get_job_status(job_id)
            print(f"Status: {status['status']}, Step: {status['step']}")
            if status['status'] in ["finished", "error", "stopped"]:
                break
            time.sleep(1)
            
        final_status = self.manager.get_job_status(job_id)
        if final_status['status'] == 'error':
            print(f"Error: {final_status.get('error')}")
            print(f"Traceback: {final_status.get('traceback')}")
            
        self.assertEqual(final_status['status'], "finished")
        self.assertGreaterEqual(final_status['step'], 4)
        
        # Check if results dir was created
        job_dir = self.manager.results_dir / job_id
        self.assertTrue(job_dir.exists())
        self.assertTrue((job_dir / "step_0000.jpg").exists())

if __name__ == '__main__':
    unittest.main()
