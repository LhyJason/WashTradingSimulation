r"""experiment.py -- run detection v4's detector over the v8 sweeps: control null first, then every run.

Run from the repo root:

  python "capstone codes/detection/v5/experiment.py" all   --jobs 15 --data <v8 data root>
  python "capstone codes/detection/v5/experiment.py" null  --jobs 15          # pass 1 only
  python "capstone codes/detection/v5/experiment.py" run   --sweep adversary --shard 0 --n-shards 8
  python "capstone codes/detection/v5/experiment.py" merge --sweep adversary

``--data`` is the folder holding ``ds_v8_<sweep>/`` (default: $WASH_V8_DATA, else the repo's
``washtrade/v8/data``).

Detection and scoring are detection v4's, unchanged: ``detect_v4`` (channels, propagation),
``fusion`` (control null, floor 0.5, exact thresholds) and ``evaluate.score_run``. Per-run work is
v4's ``experiment.detect_one``. What v5 changes is the data:

  * pass 1 (``null``) uses the control runs of the ``honest`` sweep. They cover all eight v8
    backgrounds (the four standard ones and their multi-unit ``_mu4`` versions) at the full seed
    count. The ``adversary`` sweep's control runs are the same runs (same spec, same seeds);
  * output goes to ``<data>/ds_v8_<sweep>/detection_v5/`` and ``<data>/detection_v5_{null,
    thresholds}.csv``, with the same tables as v4. ``wallets.csv`` is written for ``honest`` only.

For the four standard backgrounds the v8 control runs are the v7 control runs bit for bit
(v8 defaults == v7), so their thresholds equal detection v4's.
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from detection.v4.detector import detect_v4
from detection.v4.experiment import detect_one
from detection.v4.fusion import MAIN_FLOOR, SINGLE, Null, thresholds
from washtrade.v8 import design as D
from washtrade.v8.export import load_observable
from washtrade.v8.generate import DEFAULT_OUT as V8_DEFAULT_OUT
from washtrade.v8.generate import dataset_root

DEFAULT_DATA = os.environ.get("WASH_V8_DATA") or V8_DEFAULT_OUT
OUT_NAME = "detection_v5"
NULL_SWEEP = "honest"
WALLET_SWEEPS = ("honest",)


def out_dir(sweep, data):
    return os.path.join(dataset_root(sweep, out_dir=data), OUT_NAME)


def null_path(data):
    return os.path.join(data, f"{OUT_NAME}_null.csv")


def thresholds_path(data):
    return os.path.join(data, f"{OUT_NAME}_thresholds.csv")


# --------------------------------------------------------------------------- pass 1
def _control_pairs(arg):
    run_id, path, bg = arg
    det = detect_v4(load_observable(path)["trades"])
    p = det.pairs[["qty"] + SINGLE].copy()
    return p.assign(background=bg, run_id=run_id)


def build_null(data=DEFAULT_DATA, jobs=None, verbose=True):
    """Pass 1: control pairs of the honest sweep -> null file + thresholds table."""
    root = dataset_root(NULL_SWEEP, out_dir=data)
    man = pd.read_csv(os.path.join(root, "manifest.csv"))
    ctrl = man[man["f_scenario"] == "control"]
    args = [(r.run_id, os.path.join(root, r.path), r.background) for r in ctrl.itertuples()]
    t0 = time.time()
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    if jobs > 1:
        with Pool(jobs) as pool:
            frames = pool.map(_control_pairs, args, chunksize=8)
    else:
        frames = [_control_pairs(a) for a in args]
    control = pd.concat(frames, ignore_index=True)
    control.to_csv(null_path(data), index=False)
    th = thresholds(control, Null(control))
    th.to_csv(thresholds_path(data), index=False)
    if verbose:
        print(f"null: {len(args)} control runs, {len(control)} pairs ({time.time() - t0:.0f}s)")
        show = th[th["method"].isin(["closure", "roundtrip_100", "repeat_50", "v4"])
                  & (th["floor"] == MAIN_FLOOR)]
        print(show.pivot(index="background", columns="method", values="theta_1").round(4).to_string(),
              flush=True)
    return control, th


def load_null(data=DEFAULT_DATA):
    control = pd.read_csv(null_path(data), float_precision="round_trip")
    return Null(control), pd.read_csv(thresholds_path(data), float_precision="round_trip")


# --------------------------------------------------------------------------- pass 2
def _part(sweep, data, table, shard, n_shards):
    return os.path.join(out_dir(sweep, data), "parts", f"{table}_{shard:03d}of{n_shards:03d}.csv")


def run_shard(sweep, shard=0, n_shards=1, data=DEFAULT_DATA, force=False, verbose=True,
              null=None, th=None):
    root = dataset_root(sweep, out_dir=data)
    tables = ("results", "wallets") if sweep in WALLET_SWEEPS else ("results",)
    paths = {t: _part(sweep, data, t, shard, n_shards) for t in tables}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return 0
    if null is None:
        null, th = load_null(data)
    man = pd.read_csv(os.path.join(root, "manifest.csv"))
    factors = [c for c in man.columns if c.startswith("f_")]
    mine = man.iloc[[i for i in range(len(man)) if i % n_shards == shard]]
    res, wal = [], []
    t0 = time.time()
    for k, row in enumerate(mine.itertuples(index=False), start=1):
        ids = dict(run_id=row.run_id, cell=row.cell, seed=row.seed, background=row.background,
                   **{f: getattr(row, f) for f in factors})
        rows, w = detect_one(os.path.join(root, row.path), row.background, null, th,
                             wallets="wallets" in tables)
        res += [dict(ids, **r) for r in rows]
        if w is not None:
            wal.append(w.assign(**ids))
        if verbose and (k % 200 == 0 or k == len(mine)):
            print(f"[{sweep} shard {shard}/{n_shards}] {k}/{len(mine)} runs ({time.time() - t0:.0f}s)",
                  flush=True)
    os.makedirs(os.path.dirname(paths["results"]), exist_ok=True)
    if "wallets" in tables:
        (pd.concat(wal, ignore_index=True) if wal else pd.DataFrame()).to_csv(paths["wallets"],
                                                                             index=False)
    pd.DataFrame(res).to_csv(paths["results"], index=False)      # last: marks the shard as done
    return len(mine)


def merge(sweep, data=DEFAULT_DATA, verbose=True):
    """Concatenate the part files of the most recent sharding."""
    out = out_dir(sweep, data)
    parts = os.path.join(out, "parts")
    latest = max((f for f in os.listdir(parts) if f.startswith("results_")),
                 key=lambda f: os.path.getmtime(os.path.join(parts, f)))
    n = int(latest.rsplit("of", 1)[1][:3])
    tables = ("results", "wallets") if sweep in WALLET_SWEEPS else ("results",)
    merged = {}
    for t in tables:
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
        r = merged["results"]
        print(f"{sweep}: {r['run_id'].nunique()} runs x {r['method'].nunique()} methods merged")
    return merged


def run_all(sweeps=None, jobs=None, data=DEFAULT_DATA, force=False):
    """Pass 1, then every sweep: ``jobs`` shard processes in parallel, then merge."""
    sweeps = list(D.SWEEPS) if sweeps is None else list(sweeps)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    t_all = time.time()
    build_null(data, jobs)
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("all")
    pa.add_argument("--sweeps", nargs="+", default=None, choices=list(D.SWEEPS))
    pn = sub.add_parser("null")
    pr = sub.add_parser("run")
    pr.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    pr.add_argument("--shard", type=int, default=0)
    pr.add_argument("--n-shards", type=int, default=1)
    pm = sub.add_parser("merge")
    pm.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    for p in (pa, pn):
        p.add_argument("--jobs", type=int, default=None)
    for p in (pa, pn, pr, pm):
        p.add_argument("--data", default=DEFAULT_DATA, help="folder holding ds_v8_<sweep>/")
    for p in (pa, pr):
        p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if args.cmd == "all":
        run_all(args.sweeps, args.jobs, args.data, args.force)
    elif args.cmd == "null":
        build_null(args.data, args.jobs)
    elif args.cmd == "run":
        run_shard(args.sweep, args.shard, args.n_shards, args.data, args.force)
    else:
        merge(args.sweep, args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
