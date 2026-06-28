from pathlib import Path
import json
import datetime
import subprocess


class TrainingManager:
    def __init__(self, logs_dir: Path, runs_dir: Path, legacy_logs_dir: Path | None = None):
        self.logs_dir = logs_dir
        self.runs_dir = runs_dir
        self.legacy_logs_dir = legacy_logs_dir
        self.jobs_file = logs_dir / "training_jobs.json"
        self._ensure_dirs()
        self.jobs = self._load_jobs()
        self._purge_legacy_jobs()

    def _ensure_dirs(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _load_jobs(self):
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _save_jobs(self):
        with open(self.jobs_file, "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, indent=2)

    def _legacy_jobs_file(self) -> Path | None:
        if not self.legacy_logs_dir:
            return None
        return self.legacy_logs_dir / "training_jobs.json"

    def _load_legacy_jobs(self) -> dict:
        legacy_file = self._legacy_jobs_file()
        if not legacy_file or not legacy_file.exists():
            return {}
        try:
            with legacy_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_legacy_jobs(self, legacy_jobs: dict) -> None:
        legacy_file = self._legacy_jobs_file()
        if not legacy_file:
            return
        self.legacy_logs_dir.mkdir(parents=True, exist_ok=True)
        with legacy_file.open("w", encoding="utf-8") as f:
            json.dump(legacy_jobs, f, indent=2)

    def _is_legacy_job(self, job: dict) -> bool:
        dataset_path = str(job.get("dataset_path", "")).replace("\\", "/")
        if "/legacy_training/" in dataset_path:
            return True

        job_id = job.get("job_id") or ""
        slurm_id = job.get("slurm_id")
        if job_id and slurm_id and self.legacy_logs_dir:
            legacy_log = self.legacy_logs_dir / f"{job_id}_{slurm_id}.out"
            if legacy_log.exists():
                return True

        if self.legacy_logs_dir and dataset_path:
            ds = Path(dataset_path)
            legacy_root = self.legacy_logs_dir.parent
            legacy_ds = legacy_root / "datasets" / ds.name
            if legacy_ds.exists() and not ds.exists():
                return True
        return False

    def _purge_legacy_jobs(self) -> None:
        if not self.jobs:
            return

        legacy_jobs = self._load_legacy_jobs()
        changed = False
        for job_id in list(self.jobs.keys()):
            job = self.jobs[job_id]
            if not self._is_legacy_job(job):
                continue
            legacy_jobs[job_id] = job
            del self.jobs[job_id]
            changed = True

        if changed:
            self._save_legacy_jobs(legacy_jobs)
            self._save_jobs()

    def register_job(self, job_id, dataset_path, slurm_id=None, pid=None, model_name=None, status="submitted"):
        self.jobs[job_id] = {
            "job_id": job_id,
            "slurm_id": slurm_id,
            "pid": pid,
            "dataset_path": str(dataset_path),
            "model_name": model_name,
            "status": status,
            "submitted_at": datetime.datetime.now().isoformat(),
            "log_file": str(self.logs_dir / f"{job_id}_{slurm_id}.out") if slurm_id else None,
        }
        self._save_jobs()

    def list_jobs(self):
        # Reload from disk so external archival edits take effect without restart.
        self.jobs = self._load_jobs()
        self._purge_legacy_jobs()

        for job in self.jobs.values():
            if job["status"] in ["submitted", "running"]:
                job["status"] = self._check_status(job)
        self._save_jobs()
        return list(self.jobs.values())

    def _check_status(self, job):
        if job.get("slurm_id"):
            try:
                res = subprocess.run(["squeue", "-j", str(job["slurm_id"])], capture_output=True, text=True)
                if str(job["slurm_id"]) in res.stdout:
                    return "running"
                log_path = self.logs_dir / f"{job['job_id']}_{job['slurm_id']}.out"
                if log_path.exists():
                    content = log_path.read_text(encoding="utf-8", errors="replace")
                    if "Training finished" in content:
                        return "completed"
                    if "Error" in content or "Exception" in content:
                        return "error"
                return "completed"
            except Exception:
                return "unknown"
        return job["status"]

    def get_log(self, job_id):
        job = self.jobs.get(job_id)
        if not job:
            legacy_jobs = self._load_legacy_jobs()
            job = legacy_jobs.get(job_id)
        if not job:
            return ""

        slurm_id = job.get("slurm_id")
        if not slurm_id:
            return "Log file not found."

        search_dirs = [self.logs_dir]
        if self.legacy_logs_dir:
            search_dirs.append(self.legacy_logs_dir)

        for base in search_dirs:
            log_path = base / f"{job_id}_{slurm_id}.out"
            if log_path.exists():
                return log_path.read_text(encoding="utf-8", errors="replace")
            err_path = base / f"{job_id}_{slurm_id}.err"
            if err_path.exists():
                return err_path.read_text(encoding="utf-8", errors="replace")
        return "Log file not found."
