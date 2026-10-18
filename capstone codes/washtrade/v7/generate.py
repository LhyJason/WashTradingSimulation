r"""generate.py -- run a v7 sweep (or one shard of it) and merge the results.

Run from the repo root:

  python "capstone codes/washtrade/v7/generate.py" list
  python "capstone codes/washtrade/v7/generate.py" run   --sweep rotation --seeds 50
  python "capstone codes/washtrade/v7/generate.py" run   --sweep rotation --shard 3 --n-shards 50
  python "capstone codes/washtrade/v7/generate.py" merge --sweep rotation
  python "capstone codes/washtrade/v7/generate.py" hash  --sweep rotation   # before comparing machines
  python "capstone codes/washtrade/v7/generate.py" compare local/hashes.csv cluster/hashes.csv

On a Slurm job array, ``--shard`` / ``--n-shards`` default to SLURM_ARRAY_TASK_ID /
SLURM_ARRAY_TASK_COUNT (see ``slurm/``). Shards never write the same run folder, finished runs
(``meta.json`` present) are skipped, so a killed or re-submitted job only redoes unfinished runs.

Layout:
  <out>/<name>/                     name defaults to ds_v7_<sweep>
    design.csv                      every run of the sweep (written by any shard, identical)
    runs/<run_id>/                  observable/ labels/ meta.json  (export.py)
    errors/<run_id>.txt             traceback of a failed run (the run is retried next time)
    manifest.csv, groups.csv        written by ``merge``: one row per run / per (run, group)
    hashes.csv                      written by ``hash``: SHA-256 of every data CSV of every run,
                                    line endings normalized (Windows writes CRLF, Linux LF)
    dataset.json                    written by ``merge``
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))     # .../capstone codes
_ROOT = os.path.dirname(_CAP)                       # .../<repo root>
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch

from washtrade.v7 import design as D
from washtrade.v7 import scenarios as S
from washtrade.v7.export import SCHEMA_VERSION, export_run_atomic, is_complete, load_meta

DEFAULT_OUT = os.path.join(_HERE, "data")


def environment() -> dict:
    """Recorded in every meta.json, to compare datasets generated on different machines."""
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                pandas=pd.__version__, system=platform.system(), machine=platform.machine())


def dataset_root(sweep, name=None, out_dir=None) -> str:
    return os.path.join(out_dir or DEFAULT_OUT, name or f"ds_v7_{sweep}")


def simulate(spec: D.RunSpec):
    bg = dict(spec.background_spec, name=spec.background)
    sim = S.build(bg, spec.groups, spec.seed, sim_time=spec.sim_time,
                  record_lob=(spec.mode != "tape"))
    sim.run()
    return sim


def run_one(spec: D.RunSpec, root: str) -> float:
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


def _write_design(runs, root):
    os.makedirs(root, exist_ok=True)
    df = pd.DataFrame([dict(run_id=r.run_id, sweep=r.sweep, cell=r.cell, seed=r.seed,
                            background=r.background, mode=r.mode, sim_time=r.sim_time,
                            factors=json.dumps(r.factors, sort_keys=True),
                            background_spec=json.dumps(r.background_spec, sort_keys=True),
                            groups=json.dumps(r.groups, sort_keys=True)) for r in runs])
    text = df.to_csv(index=False)
    target = os.path.join(root, "design.csv")

    def _same():
        try:
            with open(target, encoding="utf-8", newline="") as fh:
                return fh.read() == text
        except OSError:
            return False

    if _same():
        return
    # every shard writes the same file; on Windows a concurrent os.replace onto the same target
    # can fail with PermissionError, so retry and accept another shard's identical file
    tmp = os.path.join(root, f"design.csv.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    for attempt in range(50):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if _same():
                os.remove(tmp)
                return
            time.sleep(0.05 * (attempt + 1))
    raise PermissionError(f"could not write {target}")


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
        try:
            run_one(spec, root)
            done += 1
            err = os.path.join(root, "errors", spec.run_id + ".txt")
            if os.path.exists(err):
                os.remove(err)
        except Exception:
            failed += 1
            os.makedirs(os.path.join(root, "errors"), exist_ok=True)
            with open(os.path.join(root, "errors", spec.run_id + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        if verbose and (k % 25 == 0 or k == len(mine)):
            print(f"[shard {shard}/{n_shards}] {k}/{len(mine)}  done {done}  skipped {skipped}"
                  f"  failed {failed}  ({time.time() - t0:.0f}s)", flush=True)
    return done, skipped, failed


def _flat(prefix, d):
    return {f"{prefix}{k}": v for k, v in (d or {}).items()
            if not isinstance(v, (list, dict))}


def merge(sweep, *, name=None, out_dir=None, verbose=True):
    """Collect every finished run's meta.json into manifest.csv / groups.csv."""
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
    """SHA-256 of every CSV of every finished run (meta.json is excluded: it holds timings and
    the machine description). Line endings are normalized first: pandas writes CRLF on Windows
    and LF on Linux, so raw bytes never agree across the two. Two machines produced the same
    dataset iff their hashes agree (``compare``)."""
    root = dataset_root(sweep, name, out_dir)
    rows = []
    runs_dir = os.path.join(root, "runs")
    for rid in sorted(os.listdir(runs_dir)):
        rdir = os.path.join(runs_dir, rid)
        if rid.endswith(".tmp") or not is_complete(rdir):
            continue
        for folder in ("observable", "labels"):
            for f in sorted(os.listdir(os.path.join(rdir, folder))):
                with open(os.path.join(rdir, folder, f), "rb") as fh:
                    data = fh.read().replace(b"\r\n", b"\n")
                rows.append(dict(run_id=rid, file=f"{folder}/{f}",
                                 sha256=hashlib.sha256(data).hexdigest()))
    df = pd.DataFrame(rows, columns=["run_id", "file", "sha256"])
    df.to_csv(os.path.join(root, "hashes.csv"), index=False)
    if verbose:
        print(f"hashed {df['run_id'].nunique()} runs ({len(df)} files) -> {root}/hashes.csv")
    return df


def compare_hashes(a, b, verbose=True) -> bool:
    """Compare two hashes.csv files (e.g. local vs cluster). True iff they list the same files
    with the same hashes."""
    ha, hb = pd.read_csv(a), pd.read_csv(b)
    m = ha.merge(hb, on=["run_id", "file"], how="outer", suffixes=("_a", "_b"), indicator=True)
    only_a = m[m["_merge"] == "left_only"]
    only_b = m[m["_merge"] == "right_only"]
    diff = m[(m["_merge"] == "both") & (m["sha256_a"] != m["sha256_b"])]
    same = len(m) - len(only_a) - len(only_b) - len(diff)
    if verbose:
        print(f"{same} files identical, {len(diff)} differ, {len(only_a)} only in A, "
              f"{len(only_b)} only in B  ({ha['run_id'].nunique()} vs {hb['run_id'].nunique()} runs)")
        for title, part in (("differ", diff), ("only in A", only_a), ("only in B", only_b)):
            if len(part):
                print(f"  {title} (first 10):")
                for r in part.head(10).itertuples():
                    print(f"    {r.run_id}  {r.file}")
        print("IDENTICAL" if same == len(m) and len(m) else "NOT IDENTICAL")
    return same == len(m) and len(m) > 0


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list sweeps and their sizes")

    pr = sub.add_parser("run", help="run (a shard of) a sweep")
    pr.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
    pr.add_argument("--seeds", type=int, default=D.DEFAULT_SEEDS, help="seeds 0..N-1")
    pr.add_argument("--shard", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID or 0")
    pr.add_argument("--n-shards", type=int, default=None,
                    help="default: $SLURM_ARRAY_TASK_COUNT or 1")
    pr.add_argument("--name", default=None, help="dataset folder name (default ds_v7_<sweep>)")
    pr.add_argument("--out", default=DEFAULT_OUT)
    pr.add_argument("--sim-time", type=int, default=D.DEFAULT_SIM_TIME)
    pr.add_argument("--mode", choices=["full", "slim", "tape"], default=None,
                    help="export mode (default: the sweep's own)")
    pr.add_argument("--force", action="store_true", help="redo finished runs too")
    pr.add_argument("--limit", type=int, default=None, help="only the first N runs of the shard")

    for cmd, text in (("merge", "build manifest.csv / groups.csv from finished runs"),
                      ("hash", "write hashes.csv (SHA-256 of every data file)")):
        pm = sub.add_parser(cmd, help=text)
        pm.add_argument("--sweep", required=True, choices=list(D.SWEEPS))
        pm.add_argument("--name", default=None)
        pm.add_argument("--out", default=DEFAULT_OUT)
    pc = sub.add_parser("compare", help="compare two hashes.csv files (e.g. local vs cluster)")
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
