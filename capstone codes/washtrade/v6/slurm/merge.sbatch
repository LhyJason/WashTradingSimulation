#!/bin/bash
# Merge a finished sweep (manifest.csv, groups.csv, dataset.json) and hash its data files.
# Normally submitted by submit.sh with --dependency=afterany:<array job>.
#
#SBATCH --job-name=wt-v6-merge
#SBATCH --account=CHANGE_ME
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
SWEEP="${1:?usage: merge.sbatch <sweep>}"
OUT="${DATA_ROOT:-capstone codes/washtrade/v6/data}"

cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"

python "capstone codes/washtrade/v6/generate.py" merge --sweep "$SWEEP" --out "$OUT"
python "capstone codes/washtrade/v6/generate.py" hash  --sweep "$SWEEP" --out "$OUT"
