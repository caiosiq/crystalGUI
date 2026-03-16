from pathlib import Path
import json
import shutil
import datetime
import subprocess
import re
import os

class TrainingManager:
    def __init__(self, logs_dir: Path, runs_dir: Path):
        self.logs_dir = logs_dir
        self.runs_dir = runs_dir
        self.jobs_file = logs_dir / "training_jobs.json"
        self._ensure_dirs()
        self.jobs = self._load_jobs()

    def _ensure_dirs(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_jobs(self):
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_jobs(self):
        with open(self.jobs_file, "w") as f:
            json.dump(self.jobs, f, indent=2)

    def register_job(self, job_id, dataset_path, slurm_id=None, pid=None, model_name=None, status="submitted"):
        self.jobs[job_id] = {
            "job_id": job_id,
            "slurm_id": slurm_id,
            "pid": pid,
            "dataset_path": str(dataset_path),
            "model_name": model_name,
            "status": status,
            "submitted_at": datetime.datetime.now().isoformat(),
            "log_file": str(self.logs_dir / f"{job_id}.out") if slurm_id else None
        }
        self._save_jobs()

    def list_jobs(self):
        # Update status if possible
        for jid, job in self.jobs.items():
            if job["status"] in ["submitted", "running"]:
                job["status"] = self._check_status(job)
        self._save_jobs()
        return list(self.jobs.values())

    def _check_status(self, job):
        # Simple check: if slurm_id, check squeue
        if job.get("slurm_id"):
            try:
                res = subprocess.run(["squeue", "-j", str(job["slurm_id"])], capture_output=True, text=True)
                if str(job["slurm_id"]) in res.stdout:
                    return "running"
                # If not in squeue, it's either done or failed. Check log file for "Training finished"
                log_path = Path(self.logs_dir) / f"{job['job_id']}_{job['slurm_id']}.out"
                if log_path.exists():
                    content = log_path.read_text()
                    if "Training finished" in content:
                        return "completed"
                    if "Error" in content or "Exception" in content: # heuristic
                        return "error"
                return "completed" # assume completed if not running and no obvious error
            except:
                return "unknown"
        return job["status"]

    def get_log(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            return ""
        # Try to find the log file
        # Pattern: {job_name}_{slurm_id}.out
        # job_id is usually job_name
        slurm_id = job.get("slurm_id")
        if slurm_id:
             log_path = self.logs_dir / f"{job_id}_{slurm_id}.out"
             if log_path.exists():
                 return log_path.read_text()
             err_path = self.logs_dir / f"{job_id}_{slurm_id}.err"
             if err_path.exists():
                 return err_path.read_text()
        return "Log file not found."
