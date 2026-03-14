
import threading
import uuid
import time
import cv2
import torch
import numpy as np
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

# Import DiffOSOG and Engine
# Note: Ensure PROJECT_ROOT is in sys.path before importing this module
from diff_calibration.src.diff_wrapper import DiffOSOG
from diff_calibration.src.calibration_engine import CalibrationEngine
from crystalGUI.osog.config import SynthConfig

# Global Job Store
CALIBRATION_JOBS: Dict[str, Dict[str, Any]] = {}

class CalibrationManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "data" / "results" / "calibration"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def start_job(self, 
                  target_image_path: str, 
                  initial_config: Dict[str, Any], 
                  selected_params: List[str], 
                  max_steps: int = 200,
                  learning_rate: float = 0.05,
                  device: str = "cpu") -> str:
        
        job_id = uuid.uuid4().hex
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Job State
        CALIBRATION_JOBS[job_id] = {
            "job_id": job_id,
            "status": "initializing",
            "step": 0,
            "max_steps": max_steps,
            "stage": "Pending",
            "loss": 0.0,
            "monitor_losses": {},
            "velocities": {},
            "current_params": {},
            "history": {"loss": [], "steps": []},
            "preview_url": None,
            "error": None,
            "stop_requested": False,
            "created_at": time.time()
        }

        # Spawn Worker
        thread = threading.Thread(
            target=self._worker,
            args=(job_id, job_dir, target_image_path, initial_config, selected_params, max_steps, learning_rate, device),
            daemon=True
        )
        thread.start()
        
        return job_id

    def stop_job(self, job_id: str) -> bool:
        if job_id in CALIBRATION_JOBS:
            CALIBRATION_JOBS[job_id]["stop_requested"] = True
            CALIBRATION_JOBS[job_id]["status"] = "stopping"
            return True
        return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return CALIBRATION_JOBS.get(job_id)

    def compute_loss(self,
                    target_image_path: str,
                    config: Dict[str, Any],
                    n_samples: int = 1,
                    device: str = "cpu") -> Dict[str, Any]:
        """
        Compute loss between target image and model generated with config.
        """
        try:
            # 1. Load Target Image
            if not os.path.exists(target_image_path):
                raise FileNotFoundError(f"Target image not found: {target_image_path}")
            
            target_bgr = cv2.imread(target_image_path)
            if target_bgr is None:
                raise ValueError(f"Failed to read image: {target_image_path}")
            
            if hasattr(config, 'canvas'):
                cfg = config
            else:
                cfg = SynthConfig.from_dict(config)

            h, w = cfg.canvas.height, cfg.canvas.width
            
            if target_bgr.shape[0] != h or target_bgr.shape[1] != w:
                target_bgr = cv2.resize(target_bgr, (w, h))
            
            target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
            target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).float() / 255.0
            target_tensor = target_tensor.unsqueeze(0).to(device)

            # 2. Init Engine (just for loss calc)
            diff_model = DiffOSOG(config=cfg, device=device, batch_size=1)
            
            # We don't need to register active params if we are just evaluating loss
            # But the engine init requires it? No, Engine.__init__ doesn't register params.
            # param_manager.init_parameters registers them. We skip that.
            
            engine = CalibrationEngine(
                model=diff_model,
                device=device,
                base_lr=0.0, # Irrelevant
                max_steps=0,
                rules_path=None
            )
            
            # Manually ensure losses are initialized (Engine does this in __init__)
            # Now call evaluate_loss
            with torch.no_grad():
                stats = engine.evaluate_loss(target_tensor, n_samples=n_samples)
            
            return {"ok": True, "stats": stats}
            
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}

    def _worker(self, 
                job_id: str, 
                job_dir: Path, 
                target_image_path: str, 
                initial_config: Dict[str, Any], 
                selected_params: List[str], 
                max_steps: int, 
                learning_rate: float, 
                device: str):
        
        job = CALIBRATION_JOBS[job_id]
        
        try:
            job["status"] = "running"
            
            # 1. Load Target Image
            if not os.path.exists(target_image_path):
                raise FileNotFoundError(f"Target image not found: {target_image_path}")
            
            target_bgr = cv2.imread(target_image_path)
            if target_bgr is None:
                raise ValueError(f"Failed to read image: {target_image_path}")
                
            # Resize target to config dims or update config dims?
            # DiffOSOG uses config.canvas.width/height.
            # We should probably resize target to match config, OR resize config to match target.
            # Usually optimization is faster on smaller images (e.g. 512x512).
            # For now, let's assume we resize target to match the config provided.
            
            if hasattr(initial_config, 'canvas'):
                cfg = initial_config
            else:
                cfg = SynthConfig.from_dict(initial_config)

            h, w = cfg.canvas.height, cfg.canvas.width
            
            if target_bgr.shape[0] != h or target_bgr.shape[1] != w:
                print(f"[Calibration] Resizing target from {target_bgr.shape[:2]} to ({h}, {w})")
                target_bgr = cv2.resize(target_bgr, (w, h))
            
            # Convert to Tensor (B, C, H, W) 0-1 float
            target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
            target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).float() / 255.0
            target_tensor = target_tensor.unsqueeze(0).to(device)

            # 2. Initialize Engine
            diff_model = DiffOSOG(config=cfg, device=device, batch_size=1)
            # Register params (Important! DiffOSOG needs to know what to optimize)
            # The Engine usually calls param_manager.init_parameters, but DiffOSOG needs registration first?
            # Actually, Engine.param_manager.init_parameters calls model.register_active_params.
            
            engine = CalibrationEngine(
                model=diff_model,
                device=device,
                base_lr=learning_rate,
                max_steps=max_steps,
                rules_path=None # Use default rules
            )

            # 3. Define Callback
            def progress_callback(step, total, info):
                if job["stop_requested"]:
                    raise InterruptedError("Job stopped by user")

                job["step"] = step
                job["loss"] = info.get("loss", 0.0)
                job["stage"] = info.get("stage", "Unknown")
                job["velocities"] = info.get("velocities", {})
                job["current_params"] = info.get("params", {})
                job["monitor_losses"] = info.get("monitor_losses", {})
                
                # Append history
                job["history"]["loss"].append(job["loss"])
                job["history"]["steps"].append(step)

                # Save Preview Image periodically (e.g. every 10 steps)
                if step % 10 == 0 or step == 0:
                    try:
                        # Generate current guess
                        with torch.no_grad():
                            # Use locked seed for consistent preview if desired, 
                            # but engine uses locked/random based on stage.
                            # We'll just ask the model to generate.
                            # Note: info['params'] are physical values, model is already updated.
                            # We just run forward.
                            pred_tensor = diff_model(seed=42) # Use fixed seed for preview stability
                            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                            pred_bgr = cv2.cvtColor(pred_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                            
                            # Save
                            fname = f"step_{step:04d}.jpg"
                            save_path = job_dir / fname
                            cv2.imwrite(str(save_path), pred_bgr)
                            
                            # Update URL (relative to static mount)
                            # Assuming /static/results maps to data/results
                            job["preview_url"] = f"/static/results/calibration/{job_id}/{fname}"
                    except Exception as e:
                        print(f"[Calibration] Preview generation failed: {e}")

            # 4. Run Calibration
            final_params = engine.calibrate(
                target_img=target_tensor,
                selected_params=selected_params,
                progress_callback=progress_callback
            )

            job["status"] = "finished"
            job["current_params"] = final_params
            
        except InterruptedError:
            job["status"] = "stopped"
            job["message"] = "Stopped by user"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()
            print(f"[Calibration] Job failed: {e}")
            traceback.print_exc()

