#!/bin/bash
#SBATCH -J crystal-gui
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o /home/caiosiq/chem-gui/crystalGUI/logs/gui_%j.out
#SBATCH -e /home/caiosiq/chem-gui/crystalGUI/logs/gui_%j.err

# Load modules and activate env as needed
module load python/3.10 2>/dev/null || true
module load miniforge/24.3.0-0 || true
module load cuda/12.4.0 miniforge/24.3.0-0
source ~/.bashrc 2>/dev/null || true

# Activate virtualenv if you have one
if [ -d "/home/caiosiq/chem-gui/.venv" ]; then
  source /home/caiosiq/chem-gui/.venv/bin/activate
fi

cd /home/caiosiq/chem-gui/crystalGUI
mkdir -p logs

# Optional: set CUDA device if on GPU node
export CUDA_VISIBLE_DEVICES=0

# Use gunicorn with multiple workers to utilize CPUs; each worker can serve requests concurrently
# Uvicorn workers run FastAPI app and can utilize GPU for inference if model uses CUDA
exec gunicorn -w 8 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000
