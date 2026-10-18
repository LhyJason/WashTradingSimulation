#!/bin/bash
# Check that the cluster reproduces local results, in one short job. Run from the REPO ROOT on a
# login node, after setup_env.sh:
#   ACCOUNT=<slurm account> DATA_ROOT=<scratch dir> bash "capstone codes/washtrade/v7/slurm/verify_cluster.sh"
# Inside the job: smoke test, then two small datasets with fixed names (verify_rotation = rotation
# sweep at 2 seeds, verify_evasive = evasive sweep at 1 seed), merged and hashed. Compare each
# <DATA_ROOT>/verify_<sweep>/hashes.csv with the same dataset generated locally:
#   python "capstone codes/washtrade/v7/generate.py" compare <local hashes.csv> <cluster hashes.csv>
set -euo pipefail
OUT="${DATA_ROOT:-capstone codes/washtrade/v7/data}"
HERE="capstone codes/washtrade/v7"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    ACCOUNT="${ACCOUNT:?set ACCOUNT=<slurm account>, e.g. from my_accounts}"
    exec srun --account="$ACCOUNT" --partition=standard --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --mem=8G --time=01:00:00 bash "$0"
fi

VENV="${VENV:-.venv-cluster}"
set +u    # Lmod is not safe under set -u
module load "$(cat "$VENV/PY_MODULE" 2>/dev/null || echo python/3.12)"
set -u
source "$VENV/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT"
echo "== host $(hostname), $(python --version)"
python "$HERE/run_smoke.py" > "$OUT/verify_smoke.log" 2>&1 || true
echo "== smoke: $(grep -c '\[PASS\]' "$OUT/verify_smoke.log") passed, $(grep -c '\[FAIL\]' "$OUT/verify_smoke.log") failed"
grep -E '\[FAIL\]|RESULT|Error' "$OUT/verify_smoke.log" || true
if ! grep -q "RESULT: ALL PASS" "$OUT/verify_smoke.log"; then
    echo "== FAILED: smoke test did not pass, see $OUT/verify_smoke.log"
    exit 1
fi

for spec in "rotation 2" "evasive 1"; do
    set -- $spec
    sweep=$1 seeds=$2 name="verify_$1"
    for i in 0 1 2 3; do
        python "$HERE/generate.py" run --sweep "$sweep" --seeds "$seeds" --shard "$i" --n-shards 4 \
            --out "$OUT" --name "$name" > /dev/null &
    done
    wait
    python "$HERE/generate.py" merge --sweep "$sweep" --out "$OUT" --name "$name"
    python "$HERE/generate.py" hash  --sweep "$sweep" --out "$OUT" --name "$name"
    n_err=0
    if [ -d "$OUT/$name/errors" ]; then
        n_err=$(find "$OUT/$name/errors" -type f | wc -l)
    fi
    if [ "$n_err" -gt 0 ]; then
        echo "== FAILED: $n_err runs of $name failed, see $OUT/$name/errors"
        exit 1
    fi
done
echo "== done: compare $OUT/verify_rotation/hashes.csv and $OUT/verify_evasive/hashes.csv with the local ones"
