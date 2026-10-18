#!/bin/bash
# Merge a finished sweep (manifest.csv, groups.csv, dataset.json) and hash its data files.
# Normally submitted by submit.sh with --dependency=afterany:<array job>.
#
#SBATCH --job-name=wt-v7-merge
#SBATCH --account=CHANGE_ME          # overridden by submit.sh (ACCOUNT=...)
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00          # hashing ~60k runs takes longer than merging
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
SWEEP="${1:?usage: merge.sbatch <sweep>}"
OUT="${DATA_ROOT:-capstone codes/washtrade/v7/data}"

cd "$SLURM_SUBMIT_DIR"
VENV="${VENV:-.venv-cluster}"
set +u    # Lmod is not safe under set -u
module load "$(cat "$VENV/PY_MODULE" 2>/dev/null || echo python/3.12)"
set -u
source "$VENV/bin/activate"

python "capstone codes/washtrade/v7/generate.py" merge --sweep "$SWEEP" --out "$OUT"
python "capstone codes/washtrade/v7/generate.py" hash  --sweep "$SWEEP" --out "$OUT"
