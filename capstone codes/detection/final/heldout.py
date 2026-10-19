"""Fresh-seed held-out validation of detection v6 (and v4) with FROZEN thresholds.

Generates the v8 `adversary` and `honest` sweeps for seeds SEED0..SEED0+N-1 (never used in the
design, screening or calibration, which used seeds 0..199), copies detection v6's null,
thresholds and applicability table from the original 200-seed data root unchanged, and scores the
new runs. Nothing is recalibrated.

    python heldout.py all --jobs 12 --seed0 1000 --n 100 --src $WASH_V8_DATA --out $WASH_V8_HELDOUT
"""
import argparse, os, shutil, subprocess, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SWEEPS = ("adversary", "honest")


def gen_worker(a):
    from washtrade.v8 import generate as G
    G.run_sweep(a.sweep, seeds=list(range(a.seed0, a.seed0 + a.n)), shard=a.shard,
                n_shards=a.n_shards, out_dir=a.out)


def det_worker(a):
    from detection.v6 import experiment as E
    E.run_shard(a.sweep, a.shard, a.n_shards, data=a.out)


def spawn(mode, sweep, a, logs):
    os.makedirs(logs, exist_ok=True)
    ps = []
    for i in range(a.jobs):
        fh = open(os.path.join(logs, f"{mode}_{sweep}_{i:03d}.log"), "w", encoding="utf-8")
        ps.append((subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), mode + "-worker", "--sweep", sweep,
             "--shard", str(i), "--n-shards", str(a.jobs), "--seed0", str(a.seed0), "--n", str(a.n),
             "--out", a.out], stdout=fh, stderr=subprocess.STDOUT), fh))
    codes = []
    for p, fh in ps:
        codes.append(p.wait()); fh.close()
    if any(codes):
        raise RuntimeError(f"{mode} {sweep}: shards failed {codes}, see {logs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "gen-worker", "det-worker"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--src", default=os.environ.get("WASH_V8_DATA"))
    ap.add_argument("--out", default=os.environ.get("WASH_V8_HELDOUT"))
    ap.add_argument("--sweep")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "gen-worker":
        return gen_worker(a)
    if a.cmd == "det-worker":
        return det_worker(a)
    from washtrade.v8 import generate as G
    from detection.v6 import experiment as E
    assert a.seed0 >= 200, "held-out seeds must not overlap the design seeds 0..199"
    os.makedirs(a.out, exist_ok=True)
    for f in ("detection_v6_null.csv", "detection_v6_thresholds.csv", "detection_v6_backgrounds.csv"):
        shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))     # frozen calibration
    t0 = time.time()
    for sw in SWEEPS:
        spawn("gen", sw, a, os.path.join(a.out, "logs"))
        G.merge(sw, out_dir=a.out)
        print(f"{sw} generated ({time.time() - t0:.0f}s)", flush=True)
    for sw in SWEEPS:
        spawn("det", sw, a, os.path.join(a.out, "logs"))
        E.merge(sw, data=a.out)
        print(f"{sw} detected ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
