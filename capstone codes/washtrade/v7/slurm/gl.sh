#!/bin/bash
# gl.sh -- one short command per step, for running v7 on Great Lakes from a browser shell
# (Open OnDemand), where long commands and quoting are error-prone.
#
# Put this file and wash-capstone-code.tgz in your cluster home directory, then:
#   bash ~/gl.sh help
#
# Every step prints its result and also writes it to ~/gl_logs/<step>.log, so it can be read in
# the OnDemand file viewer instead of the terminal. Long steps run in the background; their
# log ends with a line starting "== done" (or "== FAILED").
set -euo pipefail
REPO="$HOME/wash-capstone/pymarketsim-master"
V7="capstone codes/washtrade/v7"
LOGS="$HOME/gl_logs"
SELF="$(readlink -f "$0")"
mkdir -p "$LOGS"

step="${1:-help}"
if [ $# -gt 0 ]; then shift; fi

need_repo() {
    [ -d "$REPO" ] || { echo "no code at $REPO: run 'bash ~/gl.sh unpack' first"; exit 1; }
    cd "$REPO"
}
lmod() { set +u; module "$@"; set -u; }      # Lmod is not safe under set -u
background() {        # background <log name> <internal step> [args]: rerun this script detached
    local log="$LOGS/$1.log"; shift
    nohup bash "$SELF" "$@" > "$log" 2>&1 < /dev/null &
    echo "started in the background (pid $!). Check with:  tail -5 ~/gl_logs/$(basename "$log")"
}
# shards so that each array task gets about 120 runs
shards() { echo $(( ($1 * $2 + 119) / 120 )); }

case "$step" in
help)
    cat <<'EOF'
bash ~/gl.sh unpack                              unpack ~/wash-capstone-code.tgz into ~/wash-capstone
bash ~/gl.sh modules                             list Python modules (pick a 3.12.x one)
bash ~/gl.sh setup <python/3.12.x>               build .venv-cluster (background, ~5 min)
bash ~/gl.sh accounts                            Slurm accounts and scratch directories
bash ~/gl.sh verify <account> <data_root>        smoke test + verification datasets (background)
bash ~/gl.sh submit <account> <data_root> <seeds>   submit all five sweeps
bash ~/gl.sh status <data_root>                  queue + progress of every sweep
bash ~/gl.sh pack <account> <data_root>          tar all sweeps into <data_root>/ds_v7_all.tgz (background)
Logs: ~/gl_logs/<step>.log
EOF
    ;;
unpack)
    mkdir -p "$HOME/wash-capstone"
    tar -xzf "$HOME/wash-capstone-code.tgz" -C "$HOME/wash-capstone"
    { ls "$REPO"; echo "== done: code in $REPO"; } | tee "$LOGS/unpack.log"
    ;;
modules)
    { lmod -t avail python 2>&1 | grep -i '^python' || true; echo "== done"; } | tee "$LOGS/modules.log"
    ;;
setup)
    need_repo
    PY="${1:?usage: bash ~/gl.sh setup python/3.12.x}"
    background setup _setup "$PY"
    ;;
_setup)
    need_repo
    if PY_MODULE="$1" bash "$V7/slurm/setup_env.sh"; then echo "== done"; else echo "== FAILED"; fi
    ;;
accounts)
    {
        echo "== accounts"
        my_accounts 2>/dev/null || sacctmgr -n show assoc user="$USER" format=account%30
        echo "== scratch directories"
        ls -d /scratch/*/*/"$USER" 2>/dev/null || echo "(none found)"
        echo "== done"
    } 2>&1 | tee "$LOGS/accounts.log"
    ;;
verify)
    need_repo
    background verify _verify "${1:?account}" "${2:?data_root}"
    ;;
_verify)
    need_repo
    if ACCOUNT="$1" DATA_ROOT="$2" bash "$V7/slurm/verify_cluster.sh"; then
        for s in rotation evasive; do
            cp "$2/verify_$s/hashes.csv" "$LOGS/cluster_${s}_hashes.csv"
        done
        echo "== done: hashes copied to ~/gl_logs/cluster_rotation_hashes.csv and cluster_evasive_hashes.csv"
    else
        echo "== FAILED"
    fi
    ;;
submit)
    need_repo
    ACCT="${1:?account}"; ROOT="${2:?data_root}"; SEEDS="${3:?seeds}"
    mkdir -p "$ROOT"
    {
        for sc in scenarios:60 rotation:72 size:42 residual:54 evasive:292; do
            sw="${sc%%:*}"; cells="${sc##*:}"
            ACCOUNT="$ACCT" DATA_ROOT="$ROOT" bash "$V7/slurm/submit.sh" "$sw" "$SEEDS" "$(shards "$cells" "$SEEDS")"
        done
        echo "== done: submitted; follow with  bash ~/gl.sh status $ROOT"
    } 2>&1 | tee "$LOGS/submit.log"
    ;;
status)
    ROOT="${1:?data_root}"
    {
        echo "== queue: $(squeue -u "$USER" -h | wc -l) jobs ($(squeue -u "$USER" -h -t PD | wc -l) pending)"
        for d in "$ROOT"/ds_v7_*/; do
            [ -d "$d" ] || continue
            n=0; e=0           # (find on a missing folder fails, and pipefail would end the script)
            [ -d "$d/runs" ] && n=$(find "$d/runs" -mindepth 1 -maxdepth 1 -type d ! -name '*.tmp' | wc -l)
            [ -d "$d/errors" ] && e=$(find "$d/errors" -type f | wc -l)
            echo "$(basename "$d"): $n run folders, $e errors"
            if [ -f "$d/dataset.json" ]; then
                python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("   merged:", d["n_complete"], "/", d["n_design"], "missing", d["n_missing"])' "$d/dataset.json"
            fi
        done
        echo "== done"
    } 2>&1 | tee "$LOGS/status.log"
    ;;
pack)
    background pack _pack "${1:?account}" "${2:?data_root}"
    ;;
_pack)
    cd "$2"
    if srun --account="$1" --partition=standard --time=02:00:00 --mem=2G \
            tar -czf ds_v7_all.tgz ds_v7_scenarios ds_v7_rotation ds_v7_size ds_v7_residual ds_v7_evasive; then
        ls -la ds_v7_all.tgz
        echo "== done: $2/ds_v7_all.tgz"
    else
        echo "== FAILED"
    fi
    ;;
*)
    echo "unknown step '$step'; see: bash ~/gl.sh help"
    exit 1
    ;;
esac
