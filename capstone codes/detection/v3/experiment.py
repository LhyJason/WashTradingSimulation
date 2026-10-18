r"""experiment.py -- run the Sethi detector over the v7 sweeps, sharded and resumable.

Run from the repo root:

  python "capstone codes/detection/v3/experiment.py" all       --jobs 15 --data <v7 data root>
  python "capstone codes/detection/v3/experiment.py" run       --sweep rotation --shard 0 --n-shards 8
  python "capstone codes/detection/v3/experiment.py" merge     --sweep rotation
  python "capstone codes/detection/v3/experiment.py" calibrate                  # after all sweeps

``--data`` is the folder holding ``ds_v7_<sweep>/`` (default: $WASH_V7_DATA, else the repo's
``washtrade/v7/data``). Detection and scoring of one run is v2's ``detect_one``, unchanged: the
detector reads only ``observable/trades.csv`` and the horizon; labels are read afterwards.

Output: ``<data>/ds_v7_<sweep>/detection_v3/`` with v2's four tables (results, groups, curves,
wallets; see detection/v2/README.md), each row carrying the run's design factors (``f_*``), plus
``parts/`` (per-shard files; a shard whose files exist is skipped unless --force).
``calibrate`` writes ``<data>/detection_v3_thresholds.csv`` and ``calibrated.csv`` per sweep
(see calibrate.py).
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

from detection.v2.experiment import CONFIGS, detect_one
from washtrade.v7 import design as D
from washtrade.v7.generate import DEFAULT_OUT as V7_DEFAULT_OUT
from washtrade.v7.generate import dataset_root

DEFAULT_DATA = os.environ.get("WASH_V7_DATA") or V7_DEFAULT_OUT
TABLES = ("results", "groups", "curves", "wallets")
OUT_NAME = "detection_v3"


def out_dir(sweep, data):
    return os.path.join(dataset_root(sweep, out_dir=data), OUT_NAME)


def _part(sweep, data, table, shard, n_shards):
    return os.path.join(out_dir(sweep, data), "parts", f"{table}_{shard:03d}of{n_shards:03d}.csv")


def run_shard(sweep, shard=0, n_shards=1, data=DEFAULT_DATA, force=False, verbose=True,
              configs=CONFIGS):
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
        res, grp, cur, wal = detect_one(os.path.join(root, row.path), configs)
        out["results"] += [dict(ids, **r) for r in res]
        out["groups"] += [g.assign(**ids) for g in grp]
        out["curves"] += [c.assign(**ids) for c in cur]
        out["wallets"] += [w.assign(**ids) for w in wal]
        if verbose and (k % 200 == 0 or k == len(mine)):
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


def merge(sweep, data=DEFAULT_DATA, verbose=True):
    """Concatenate the part files of the most recent sharding into one file per table."""
    out = out_dir(sweep, data)
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


def run_all(sweeps=None, jobs=None, data=DEFAULT_DATA, force=False):
    """Every sweep: ``jobs`` shard processes in parallel, then merge."""
    sweeps = list(D.SWEEPS) if sweeps is None else list(sweeps)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    t_all = time.time()
    for sw in sweeps:
        t0 = time.time()
        log_dir = os.path.join(out_dir(sw, data), "logs")
        os.makedirs(log_dir, exist_ok=True)
        procs = []
        for i in range(jobs):
            fh = open(os.path.join(log_dir, f"shard{i:03d}.log"), "w", encoding="utf-8")
            procs.append((subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "run", "--sweep", sw, "--shard", str(i),
                 "--n-shards", str(jobs), "--data", data] + (["--force"] if force else []),
                stdout=fh, stderr=subprocess.STDOUT), fh))
        codes = []
        for p, fh in procs:
            codes.append(p.wait())
            fh.close()
        if any(codes):
            raise RuntimeError(f"{sw}: {sum(c != 0 for c in codes)} shard(s) failed, see {log_dir}")
        merge(sw, data, verbose=False)
        print(f"{sw:10s} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"all sweeps in {(time.time() - t_all) / 60:.1f} min")


def main(argv=None):
    from detection.v3 import calibrate as C

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
    pc = sub.add_parser("calibrate")
    pc.add_argument("--fpr", type=float, default=C.TARGET_FPR)
    for p in (pa, pr, pm, pc):
        p.add_argument("--data", default=DEFAULT_DATA, help="folder holding ds_v7_<sweep>/")
    for p in (pa, pr):
        p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.cmd == "all":
        run_all(args.sweeps, args.jobs, args.data, args.force)
        C.calibrate_all(args.data)
    elif args.cmd == "run":
        run_shard(args.sweep, args.shard, args.n_shards, args.data, args.force)
    elif args.cmd == "merge":
        merge(args.sweep, args.data)
    else:
        C.calibrate_all(args.data, target=args.fpr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
