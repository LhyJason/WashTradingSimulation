r"""experiment.py -- run the Sethi detector over v6 sweeps, sharded and resumable.

Run from the repo root:

  python "capstone codes/detection/v2/experiment.py" all   --jobs 8          # every sweep, then merge
  python "capstone codes/detection/v2/experiment.py" run   --sweep gap --shard 0 --n-shards 8
  python "capstone codes/detection/v2/experiment.py" merge --sweep gap

Input: a v6 dataset ``<data>/ds_v6_<sweep>/`` (manifest.csv + runs/). The detector reads only
``observable/trades.csv`` and the horizon from meta.json; labels are read afterwards to score it.

Output: ``<data>/ds_v6_<sweep>/detection_v2/``
  results.csv   one row per (config, run): v1's metrics + same-group metrics + design factors
  groups.csv    one row per (config, run, wash group): leakage, outsider score, x_bar vs x_pred
  curves.csv    one row per (config, run, theta in 0.50..1.00): true/false positive counts
  wallets.csv   one row per (run, wallet), paper config: closures, random-walk prediction, scores,
                truth (see closing.py)
  parts/        per-shard files (a shard whose files exist is skipped unless --force)

Configurations:
  paper           Algorithms 1 + 2 as in the paper (theta = 0.9; theta* searched in [0.8, 0.99])
  no_propagation  ablation: final score = x0 (no network step)
  theta_hi_1      Algorithm 2 searched in [0.8, 1.0]; same scores as ``paper``, so no curves
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from detection.v1.pipeline import DetectorConfig, detect
from detection.v2.closing import wallet_table
from detection.v2.evaluate import evaluate, group_table, threshold_curve
from washtrade.v6 import design as D
from washtrade.v6.export import load_labels, load_meta, load_observable
from washtrade.v6.generate import DEFAULT_OUT, dataset_root

CONFIGS = {
    "paper":          DetectorConfig(),
    "no_propagation": DetectorConfig(propagate=False),
    "theta_hi_1":     DetectorConfig(theta_hi=1.0),
}
CURVE_CONFIGS = ("paper", "no_propagation")
TABLES = ("results", "groups", "curves", "wallets")
WALLET_CONFIG = "paper"


def _out_dir(sweep, data):
    return os.path.join(dataset_root(sweep, out_dir=data), "detection_v2")


def _part(sweep, data, table, shard, n_shards):
    return os.path.join(_out_dir(sweep, data), "parts", f"{table}_{shard:03d}of{n_shards:03d}.csv")


def detect_one(run_dir, configs=CONFIGS):
    """Detector + evaluation on one run. Returns (result rows, group, curve and wallet frames)."""
    trades = load_observable(run_dir)["trades"]
    horizon = load_meta(run_dir)["config"]["sim_time"]
    labels = load_labels(run_dir)
    res, grp, cur, wal = [], [], [], []
    for name, cfg in configs.items():
        det = detect(trades, cfg, horizon=horizon)
        res.append(dict(config=name, **evaluate(det, labels)))
        grp.append(group_table(det, labels).assign(config=name))
        if name in CURVE_CONFIGS:
            cur.append(threshold_curve(det, labels).assign(config=name))
        if name == WALLET_CONFIG:
            wal.append(wallet_table(det, trades, labels))
    return res, grp, cur, wal


def run_shard(sweep, shard=0, n_shards=1, data=DEFAULT_OUT, force=False, verbose=True):
    """Detect every run of this shard and write its part files. Returns the number of runs."""
    root = dataset_root(sweep, out_dir=data)
    paths = {t: _part(sweep, data, t, shard, n_shards) for t in TABLES}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return 0
    man = pd.read_csv(os.path.join(root, "manifest.csv"))
    factors = [c for c in man.columns if c.startswith("f_")]
    mine = man.iloc[[i for i in range(len(man)) if i % n_shards == shard]]
    out = {t: [] for t in TABLES}
    t0 = time.time()
    for k, row in enumerate(mine.itertuples(index=False), start=1):
        ids = dict(run_id=row.run_id, cell=row.cell, seed=row.seed, background=row.background,
                   **{f: getattr(row, f) for f in factors})
        res, grp, cur, wal = detect_one(os.path.join(root, row.path))
        out["results"] += [dict(ids, **r) for r in res]
        out["groups"] += [g.assign(**ids) for g in grp]
        out["curves"] += [c.assign(**ids) for c in cur]
        out["wallets"] += [w.assign(**ids) for w in wal]
        if verbose and (k % 100 == 0 or k == len(mine)):
            print(f"[{sweep} shard {shard}/{n_shards}] {k}/{len(mine)} runs ({time.time() - t0:.0f}s)",
                  flush=True)
    os.makedirs(os.path.dirname(paths["results"]), exist_ok=True)
    frames = {t: (pd.concat(out[t], ignore_index=True) if out[t] else pd.DataFrame())
              for t in TABLES if t != "results"}
    frames["results"] = pd.DataFrame(out["results"])
    for t in TABLES:                                   # results last: it marks the shard as done
        if t != "results":
            frames[t].to_csv(paths[t], index=False)
    frames["results"].to_csv(paths["results"], index=False)
    return len(mine)


def merge(sweep, data=DEFAULT_OUT, verbose=True):
    """Concatenate the part files of the most recent sharding into results/groups/curves.csv."""
    out = _out_dir(sweep, data)
    parts = os.path.join(out, "parts")
    latest = max((f for f in os.listdir(parts) if f.startswith("results_")),
                 key=lambda f: os.path.getmtime(os.path.join(parts, f)))
    n = int(latest.rsplit("of", 1)[1][:3])
    merged = {}
    for t in TABLES:
        frames = []
        for i in range(n):
            p = os.path.join(parts, f"{t}_{i:03d}of{n:03d}.csv")
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing shard file {os.path.basename(p)}")
            try:
                frames.append(pd.read_csv(p))
            except pd.errors.EmptyDataError:
                pass
        merged[t] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        merged[t].to_csv(os.path.join(out, f"{t}.csv"), index=False)
    if verbose:
        n_runs = merged["results"]["run_id"].nunique() if len(merged["results"]) else 0
        print(f"{sweep}: {n_runs} runs x {merged['results']['config'].nunique()} configs merged")
    return merged


def run_all(sweeps=None, jobs=None, data=DEFAULT_OUT, force=False):
    """Every sweep: ``jobs`` shard processes in parallel, then merge."""
    sweeps = list(D.SWEEPS) if sweeps is None else list(sweeps)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    t_all = time.time()
    for sw in sweeps:
        t0 = time.time()
        procs = [subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run", "--sweep", sw, "--shard", str(i),
             "--n-shards", str(jobs), "--data", data] + (["--force"] if force else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for i in range(jobs)]
        errs = [p.communicate()[1] for p in procs]
        bad = [e for p, e in zip(procs, errs) if p.returncode != 0]
        if bad:
            raise RuntimeError(f"{sw}: {len(bad)} shard(s) failed:\n{bad[0][-2000:]}")
        merge(sw, data, verbose=False)
        print(f"{sw:10s} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"all sweeps in {(time.time() - t_all) / 60:.1f} min")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("all")
    pa.add_argument("--sweeps", nargs="+", default=None, choices=list(D.SWEEPS))
    pa.add_argument("--jobs", type=int, default=None)
    pr = sub.add_parser("run")
    pr.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    pr.add_argument("--shard", type=int, default=0)
    pr.add_argument("--n-shards", type=int, default=1)
    pm = sub.add_parser("merge")
    pm.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    for p in (pa, pr, pm):
        p.add_argument("--data", default=DEFAULT_OUT, help="folder holding ds_v6_<sweep>/")
    for p in (pa, pr):
        p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.cmd == "all":
        run_all(args.sweeps, args.jobs, args.data, args.force)
    elif args.cmd == "run":
        run_shard(args.sweep, args.shard, args.n_shards, args.data, args.force)
    else:
        merge(args.sweep, args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
