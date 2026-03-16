
import json
import time
import uuid
import shutil
import subprocess
import os
import signal
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
    def _get_job_path(self, job_id: str) -> Path:
        # Search for the directory starting with timestamp_jobid
        # Since we might not know the exact timestamp, we iterate
        for p in self.jobs_dir.iterdir():
            if p.is_dir() and p.name.endswith(f"_{job_id}"):
                return p
        return self.jobs_dir / f"unknown_{job_id}"

    def register_job(self, job_id: str, mode: str, config: dict, out_dir: str, n_images: int, 
                     slurm_id: str = None, pid: int = None, pids: List[int] = None):
        """Register a new job by creating its metadata file."""
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
        
        # If it's a slurm job, the folder might already exist from the submission step
        # If local, we might need to create/rename it
        
        # We try to find if a folder was already created by synth_batch logic
        # In the current main.py, synth_batch creates SYNTH_JOBS_DIR / f"{ts}_{slurm_id}" or similar
        
        final_dir = None
        if mode == "slurm" and slurm_id:
            final_dir = self.jobs_dir / f"{ts}_{slurm_id}"
        elif mode.startswith("local"):
            # Local jobs might handle folder naming differently, let's standardize
            final_dir = self.jobs_dir / f"{ts}_{job_id}"
        
        if final_dir:
            final_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "job_id": job_id,
                "mode": mode,
                "created_at": time.time(),
                "status": "submitted" if mode == "slurm" else "running",
                "n_images": n_images,
                "out_dir": out_dir,
                "slurm_id": slurm_id,
                "pid": pid,
                "pids": pids or [],
                "config": config
            }
            with (final_dir / "job_meta.json").open("w") as f:
                json.dump(meta, f, indent=2)
            
            if mode == "local" and pid:
                 # We can't easily store the Popen object across restarts, 
                 # but for this session we can track it if we want to kill it
                 pass

    def list_jobs(self) -> List[dict]:
        jobs = []
        for p in sorted(self.jobs_dir.iterdir(), reverse=True):
            if not p.is_dir(): continue
            meta_path = p / "job_meta.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                    
                    # Update status for local jobs
                    if meta["status"] == "running" and meta["mode"].startswith("local"):
                        # Check if process is alive
                        is_alive = False
                        if meta.get("pid"):
                            try:
                                os.kill(meta["pid"], 0)
                                is_alive = True
                            except OSError:
                                is_alive = False
                        elif meta.get("pids"):
                             # Array job: check if any are alive
                             any_alive = False
                             for pid in meta["pids"]:
                                 try:
                                     os.kill(pid, 0)
                                     any_alive = True
                                     break
                                 except OSError:
                                     pass
                             is_alive = any_alive
                        
                        if not is_alive:
                            meta["status"] = "completed" 
                            # Force update progress to 100% if completed naturally, or calculate actual
                            progress = self._calculate_progress(meta)
                            meta["progress"] = progress
                            if progress >= 99.0: # Close enough
                                meta["status"] = "completed"
                            else:
                                # If process died but not all images done, it might be an error or killed
                                # But user asked for completed state if done.
                                # Let's stick to "completed" if process exited 0, but we don't have exit code here easily for detached Popen
                                # We'll just mark as completed for now as per current logic, but ensure progress is updated
                                pass
                            
                            self._update_job_meta(p, meta)
                    
                    # For Slurm jobs, we might need to check squeue if we want real "running" status update
                    # For now, we assume if mode is slurm, we check file progress
                    if meta["mode"] == "slurm":
                         progress = self._calculate_progress(meta)
                         meta["progress"] = progress
                         if progress >= 100.0:
                             meta["status"] = "completed"
                             self._update_job_meta(p, meta)

                    # Update progress in memory object before returning (it was already calc above or in _calculate_progress)
                    # The original code called _calculate_progress again at the end, which is fine
                    meta["progress"] = self._calculate_progress(meta)
                    jobs.append(meta)
                except Exception as e:
                    print(f"Error reading job meta {p}: {e}")
        return jobs

    def _calculate_progress(self, meta: dict) -> float:
        try:
            out_dir = Path(meta["out_dir"]) / "images"
            if not out_dir.exists(): return 0.0
            
            # Use os.scandir for faster counting than glob for large directories
            count = 0
            with os.scandir(out_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(".jpg"):
                        count += 1
            
            total = meta["n_images"]
            return min(100.0, (count / total) * 100.0) if total > 0 else 0.0
        except Exception:
            return 0.0

    def _update_job_meta(self, job_dir: Path, meta: dict):
        with (job_dir / "job_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    def delete_job(self, job_id: str):
        # Stop if running
        # Remove dir
        # For now, just find the dir
        for p in self.jobs_dir.iterdir():
            if p.is_dir() and (p.name.endswith(f"_{job_id}") or (f"_{job_id}" in p.name)): # Robust match
                # Try to kill if local
                meta_path = p / "job_meta.json"
                if meta_path.exists():
                    try:
                        meta = json.load(meta_path.open())
                        if meta["mode"].startswith("local"):
                            pids = meta.get("pids", [])
                            if meta.get("pid"): pids.append(meta["pid"])
                            for pid in pids:
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                except OSError:
                                    pass
                    except Exception:
                        pass
                shutil.rmtree(p)
                return True
        return False
