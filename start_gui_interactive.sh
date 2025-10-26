#!/usr/bin/env bash
set -euo pipefail

# Interactive Slurm GUI launcher
# Usage example:
#   ./start_gui_interactive.sh -p mit_preemptable -g gpu:h200:1 -c 8 -m 128G -t 06:00:00 -w 4 -P 8000 -e /home/caiosiq/chem-gui/.venv
# Defaults (override with flags):
#   partition=mit_preemptable, gres=gpu:h200:1, cpus=8, mem=128G, time=06:00:00, workers=4, port=8000, venv=/home/caiosiq/chem-gui/.venv

PART="mit_preemptable"
GRES="gpu:h200:1"
CPUS="8"
MEM="128G"
TIME="06:00:00"
WORKERS="4"
PORT="8000"
VENV="/home/caiosiq/chem-gui/.venv"

usage() {
  echo "Usage: $0 [-p partition] [-g gres] [-c cpus] [-m mem] [-t time] [-w workers] [-P port] [-e venv]" >&2
}

while getopts ":p:g:c:m:t:w:P:e:h" opt; do
  case "$opt" in
    p) PART="$OPTARG" ;;
    g) GRES="$OPTARG" ;;
    c) CPUS="$OPTARG" ;;
    m) MEM="$OPTARG" ;;
    t) TIME="$OPTARG" ;;
    w) WORKERS="$OPTARG" ;;
    P) PORT="$OPTARG" ;;
    e) VENV="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

SRUN_ARGS="-p $PART -c $CPUS --mem=$MEM --time=$TIME"
if [ -n "$GRES" ]; then
  SRUN_ARGS="$SRUN_ARGS --gres=$GRES"
fi

echo "Requesting interactive allocation: $SRUN_ARGS"

# Attach an interactive shell on the compute node and start the GUI server
srun --pty $SRUN_ARGS bash -lc "\
  set -e; \
  echo 'Compute node:' \$(hostname); \
  module load python/3.10 2>/dev/null || true; \
  module load cuda/12.2 2>/dev/null || true; \
  source ~/.bashrc 2>/dev/null || true; \
  if [ -d '$VENV' ]; then source '$VENV/bin/activate'; fi; \
  cd /home/caiosiq/chem-gui/crystalGUI; \
  # Ensure dependencies exist (uvicorn, fastapi, etc.) \
  if ! command -v uvicorn >/dev/null 2>&1; then \
    echo 'Installing crystalGUI requirements...'; \
    python -m pip install -r requirements.txt; \
  fi; \
  export CUDA_VISIBLE_DEVICES=0; \
  echo 'Starting uvicorn with' $WORKERS 'workers on port' $PORT; \
  echo 'Tunnel from your laptop (replace <login_host>):'; \
  echo '  ssh -L ' $PORT ':' \$(hostname) ':' $PORT ' caiosiq@<login_host>'; \
  uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers $WORKERS \
"
