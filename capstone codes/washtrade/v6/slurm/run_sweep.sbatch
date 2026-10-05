#!/bin/bash
# One shard of a v6 sweep per array task. Submit from the REPO ROOT, normally via submit.sh:
#   sbatch --array=0-49 "capstone codes/washtrade/v6/slurm/run_sweep.sbatch" <sweep> [seeds]
# Shard i of N = runs with index % N == i (generate.py reads SLURM_ARRAY_TASK_ID / _COUNT),
# so the array must be 0..N-1. Finished runs are skipped: re-submitting only redoes the rest.
#
#SBATCH --job-name=wt-v6
#SBATCH --account=CHANGE_ME
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
SWEEP="${1:?usage: run_sweep.sbatch <sweep> [seeds]}"
SEEDS="${2:-20}"

cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # one core per task; no thread oversubscription

python "capstone codes/washtrade/v6/generate.py" run \
    --sweep "$SWEEP" --seeds "$SEEDS" --out "${DATA_ROOT:-capstone codes/washtrade/v6/data}"
