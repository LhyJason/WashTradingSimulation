#!/bin/bash
# Submit a whole sweep: a job array of shards, then a merge job that runs when the array ends.
# Run from the REPO ROOT:
#   ACCOUNT=<slurm account> bash "capstone codes/washtrade/v7/slurm/submit.sh" <sweep> [seeds=50] [n_shards=50]
# ACCOUNT (required) is passed to sbatch and overrides the CHANGE_ME placeholder in the .sbatch files.
# Optional environment: VENV (default .venv-cluster), DATA_ROOT (default: v7/data in the repo).
set -euo pipefail
SWEEP="${1:?usage: ACCOUNT=<acct> submit.sh <sweep> [seeds] [n_shards]}"
SEEDS="${2:-50}"
N="${3:-50}"
ACCOUNT="${ACCOUNT:?set ACCOUNT=<slurm account>, e.g. from my_accounts}"
HERE="capstone codes/washtrade/v7/slurm"

mkdir -p logs
jid=$(sbatch --parsable --account="$ACCOUNT" --array=0-$((N - 1)) "$HERE/run_sweep.sbatch" "$SWEEP" "$SEEDS")
mid=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterany:"$jid" "$HERE/merge.sbatch" "$SWEEP")
echo "sweep $SWEEP: array job $jid ($N shards, $SEEDS seeds), merge job $mid"
