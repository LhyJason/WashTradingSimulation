#!/bin/bash
# Submit a whole sweep: a job array of shards, then a merge job that runs when the array ends.
# Run from the REPO ROOT:
#   bash "capstone codes/washtrade/v6/slurm/submit.sh" <sweep> [seeds=20] [n_shards=50]
# Optional environment: VENV (default .venv-cluster), DATA_ROOT (default: v6/data in the repo).
set -euo pipefail
SWEEP="${1:?usage: submit.sh <sweep> [seeds] [n_shards]}"
SEEDS="${2:-20}"
N="${3:-50}"
HERE="capstone codes/washtrade/v6/slurm"

mkdir -p logs
jid=$(sbatch --parsable --array=0-$((N - 1)) "$HERE/run_sweep.sbatch" "$SWEEP" "$SEEDS")
mid=$(sbatch --parsable --dependency=afterany:"$jid" "$HERE/merge.sbatch" "$SWEEP")
echo "sweep $SWEEP: array job $jid ($N shards, $SEEDS seeds), merge job $mid"
