#!/bin/bash
#SBATCH --job-name=crystal_synth
#SBATCH --output=/orcd/home/002/caiosiq/chem-gui/crystalGUI/data/synth_jobs/2025_10_25_20_57_%A/job.out
#SBATCH --error=/orcd/home/002/caiosiq/chem-gui/crystalGUI/data/synth_jobs/2025_10_25_20_57_%A/job.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-249
#SBATCH --partition=mit_preemptable


module purge
module load python || true

cd /orcd/home/002/caiosiq/chem-gui
export PYTHONPATH=/orcd/home/002/caiosiq/chem-gui

# Limit library threads to avoid oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_TOTAL=20000
SEED_BASE=123456
INDEX_OFFSET_BASE=0
N_TASKS=250
SHARD_SIZE=$(( (N_TOTAL + N_TASKS - 1) / N_TASKS ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_I=$(( TASK_ID * SHARD_SIZE ))
REMAIN=$(( N_TOTAL - START_I ))
N_THIS=$(( REMAIN < SHARD_SIZE ? REMAIN : SHARD_SIZE ))
OFFSET=$(( INDEX_OFFSET_BASE + START_I ))

echo "Starting batch synth task $TASK_ID of $N_TASKS: images=$N_THIS, offset=$OFFSET"
python -m crystalGUI.data_generator.batch_job --n-images "$N_THIS" --out-dir "/home/caiosiq/chem-gui/synth_speckles/gui-generated" --config-file "/orcd/home/002/caiosiq/chem-gui/crystalGUI/data/synth_jobs/2025_10_25_20_57_staging/config.json" --seed-base "$SEED_BASE" --index-offset "$OFFSET"
echo "Task finished"
