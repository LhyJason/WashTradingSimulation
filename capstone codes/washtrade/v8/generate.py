r"""generate.py -- run a v8 sweep (or one shard of it) and merge the results.

Run from the repo root:

  python "capstone codes/washtrade/v8/generate.py" list
  python "capstone codes/washtrade/v8/generate.py" all   --jobs 15 --out <data root>   # every sweep
  python "capstone codes/washtrade/v8/generate.py" run   --sweep adversary --shard 3 --n-shards 15
  python "capstone codes/washtrade/v8/generate.py" merge --sweep adversary
  python "capstone codes/washtrade/v8/generate.py" hash  --sweep adversary

Same layout and resume rules as v7 (``washtrade/v7/generate.py``): ``<out>/ds_v8_<sweep>/`` with
design.csv, runs/<run_id>/ (observable/, labels/, meta.json), errors/, and, after ``merge``,
manifest.csv, groups.csv and dataset.json. Finished runs are skipped. ``hash`` and ``compare`` are
v7's (line endings normalized). ``all`` runs every sweep in ``--jobs`` parallel shard processes,
then merges each.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from washtrade.v7.generate import _flat, _write_design, compare_hashes, environment
from washtrade.v7.generate import hash_runs as _v7_hash_runs
from washtrade.v8 import design as D
from washtrade.v8 import scenarios as S
from washtrade.v8.export import SCHEMA_VERSION, export_run_atomic, is_complete, load_meta

DEFAULT_OUT = os.environ.get("WASH_V8_DATA") or os.path.join(_HERE, "data")


def dataset_root(sweep, name=None, out_dir=None) -> str:
    return os.path.join(out_dir or DEFAULT_OUT, name or f"ds_v8_{sweep}")


def simulate(spec):
    bg = dict(spec.background_spec, name=spec.background)
    sim = S.build(bg, spec.groups, spec.seed, sim_time=spec.sim_time,
                  record_lob=(spec.mode != "tape"))
    sim.run()
    return sim


def run_one(spec, root) -> float:
    rdir = os.path.join(root, "runs", spec.run_id)
    t0 = time.time()
    sim = simulate(spec)
    elapsed = time.time() - t0
    export_run_atomic(
        sim, rdir, run_id=spec.run_id, seed=spec.seed, scenario=spec.cell,
        background=spec.background, background_spec=spec.background_spec,
        factors=spec.factors, mode=spec.mode,
        extra_meta=dict(sweep=spec.sweep, cell=spec.cell, sim_seconds=round(elapsed, 3),
                        env=environment()))
    return elapsed


def run_sweep(sweep, *, seeds=D.DEFAULT_SEEDS, shard=0, n_shards=1, name=None, out_dir=None,
              sim_time=D.DEFAULT_SIM_TIME, mode=None, force=False, limit=None, verbose=True):
    """Run this shard's unfinished runs. Returns (n_done, n_skipped, n_failed)."""
    runs = D.enumerate_runs(sweep, seeds=seeds, sim_time=sim_time, mode=mode)
    root = dataset_root(sweep, name, out_dir)
    _write_design(runs, root)
    mine = D.shard(runs, shard, n_shards)
    if limit is not None:
        mine = mine[:limit]
    done = skipped = failed = 0
    t0 = time.time()
    for k, spec in enumerate(mine, start=1):
        rdir = os.path.join(root, "runs", spec.run_id)
        if not force and is_complete(rdir):
            skipped += 1
            continue
        err = os.path.join(root, "errors", spec.run_id + ".txt")
        try:
            run_one(spec, root)
            done += 1
            if os.path.exists(err):
                os.remove(err)
        except Exception:
            failed += 1
            os.makedirs(os.path.dirname(err), exist_ok=True)
            with open(err, "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        if verbose and (k % 100 == 0 or k == len(mine)):
            print(f"[{sweep} shard {shard}/{n_shards}] {k}/{len(mine)}  done {done}  skipped "
                  f"{skipped}  failed {failed}  ({time.time() - t0:.0f}s)", flush=True)
    return done, skipped, failed


def merge(sweep, *, name=None, out_dir=None, verbose=True):
    """Collect every finished run's meta.json into manifest.csv / groups.csv (as v7)."""
    root = dataset_root(sweep, name, out_dir)
    design = pd.read_csv(os.path.join(root, "design.csv"))
    rows, grows = [], []
    for rid in design["run_id"]:
        rdir = os.path.join(root, "runs", rid)
        if not is_complete(rdir):
            continue
        m = load_meta(rdir)
        base = dict(run_id=rid, sweep=m.get("sweep"), cell=m.get("cell"), seed=m["seed"],
                    background=m.get("background"), mode=m.get("export_mode"),
                    path=os.path.relpath(rdir, root).replace(os.sep, "/"),
                    sim_seconds=m.get("sim_seconds"))
        rows.append({**base, **_flat("f_", m.get("factors")), **_flat("", m.get("summary"))})
        for g in m.get("groups", []):
            grows.append({"run_id": rid, "cell": m.get("cell"), "seed": m["seed"],
                          **_flat("f_", m.get("factors")), **_flat("", g)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(root, "manifest.csv"), index=False)
    pd.DataFrame(grows).to_csv(os.path.join(root, "groups.csv"), index=False)
    missing = sorted(set(design["run_id"]) - set(manifest.get("run_id", [])))
    with open(os.path.join(root, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(name=os.path.basename(root), sweep=sweep, schema_version=SCHEMA_VERSION,
                       merged=datetime.now().isoformat(timespec="seconds"),
                       n_design=int(len(design)), n_complete=int(len(manifest)),
                       n_missing=len(missing), missing_examples=missing[:20],
                       backgrounds_spec=S.BACKGROUNDS, env=environment()),
                  fh, indent=2, default=str)
    if verbose:
        print(f"merged {len(manifest)}/{len(design)} runs into {root}"
              + (f"  ({len(missing)} missing)" if missing else ""))
    return manifest


def hash_runs(sweep, *, name=None, out_dir=None, verbose=True):
    return _v7_hash_runs(sweep, name=name or f"ds_v8_{sweep}", out_dir=out_dir or DEFAULT_OUT,
                         verbose=verbose)


def run_all(sweeps=None, *, seeds=D.DEFAULT_SEEDS, jobs=None, out_dir=None):
    """Every sweep: ``jobs`` shard processes in parallel, then merge. Logs in <dataset>/logs."""
    sweeps = list(D.SWEEPS) if sweeps is None else list(sweeps)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    out_dir = out_dir or DEFAULT_OUT
    t_all = time.time()
    for sw in sweeps:
        t0 = time.time()
        log_dir = os.path.join(dataset_root(sw, out_dir=out_dir), "logs")
        os.makedirs(log_dir, exist_ok=True)
        procs = []
        for i in range(jobs):
            fh = open(os.path.join(log_dir, f"shard{i:03d}.log"), "w", encoding="utf-8")
            procs.append((subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "run", "--sweep", sw, "--seeds",
                 str(seeds), "--shard", str(i), "--n-shards", str(jobs), "--out", out_dir],
                stdout=fh, stderr=subprocess.STDOUT), fh))
        codes = []
        for p, fh in procs:
            codes.append(p.wait())
            fh.close()
        merge(sw, out_dir=out_dir, verbose=True)
        if any(codes):
            print(f"{sw}: {sum(c != 0 for c in codes)} shard(s) reported failed runs; see "
                  f"{os.path.join(dataset_root(sw, out_dir=out_dir), 'errors')}", flush=True)
        print(f"{sw:10s} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"all sweeps in {(time.time() - t_all) / 60:.1f} min")


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list sweeps and their sizes")

    pa = sub.add_parser("all", help="every sweep in parallel shards, then merge")
    pa.add_argument("--sweeps", nargs="+", default=None, choices=list(D.SWEEPS))
    pa.add_argument("--seeds", type=int, default=D.DEFAULT_SEEDS)
    pa.add_argument("--jobs", type=int, default=None)
    pa.add_argument("--out", default=DEFAULT_OUT)

    pr = sub.add_parser("run", help="run (a shard of) a sweep")
    pr.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    pr.add_argument("--seeds", type=int, default=D.DEFAULT_SEEDS, help="seeds 0..N-1")
    pr.add_argument("--shard", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID or 0")
    pr.add_argument("--n-shards", type=int, default=None,
                    help="default: $SLURM_ARRAY_TASK_COUNT or 1")
    pr.add_argument("--name", default=None, help="dataset folder name (default ds_v8_<sweep>)")
    pr.add_argument("--out", default=DEFAULT_OUT)
    pr.add_argument("--sim-time", type=int, default=D.DEFAULT_SIM_TIME)
    pr.add_argument("--mode", choices=["full", "slim", "tape"], default=None)
    pr.add_argument("--force", action="store_true", help="redo finished runs too")
    pr.add_argument("--limit", type=int, default=None, help="only the first N runs of the shard")

    for cmd, text in (("merge", "build manifest.csv / groups.csv from finished runs"),
                      ("hash", "write hashes.csv (SHA-256 of every data file)")):
        pm = sub.add_parser(cmd, help=text)
        pm.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
        pm.add_argument("--name", default=None)
        pm.add_argument("--out", default=DEFAULT_OUT)
    pc = sub.add_parser("compare", help="compare two hashes.csv files")
    pc.add_argument("a")
    pc.add_argument("b")

    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.cmd == "list":
        for sw in D.SWEEPS:
            runs = D.enumerate_runs(sw, seeds=1)
            print(f"  {sw:10s} {len(runs):5d} cells  x seeds  (export: {runs[0].mode})")
        return 0
    if args.cmd == "all":
        run_all(args.sweeps, seeds=args.seeds, jobs=args.jobs, out_dir=args.out)
        return 0
    if args.cmd == "run":
        shard = args.shard if args.shard is not None else _env_int("SLURM_ARRAY_TASK_ID", 0)
        n_shards = (args.n_shards if args.n_shards is not None
                    else _env_int("SLURM_ARRAY_TASK_COUNT", 1))
        _, _, failed = run_sweep(args.sweep, seeds=args.seeds, shard=shard, n_shards=n_shards,
                                 name=args.name, out_dir=args.out, sim_time=args.sim_time,
                                 mode=args.mode, force=args.force, limit=args.limit)
        return 1 if failed else 0
    if args.cmd == "hash":
        hash_runs(args.sweep, name=args.name, out_dir=args.out)
        return 0
    if args.cmd == "compare":
        return 0 if compare_hashes(args.a, args.b) else 1
    merge(args.sweep, name=args.name, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
