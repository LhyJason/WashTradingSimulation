r"""run_smoke.py -- detection v5 smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root (generates its own small v7 and v8 datasets in a temporary folder):
  python "capstone codes/detection/v5/run_smoke.py"

  (A) null           built from the honest sweep's control runs: all eight v8 backgrounds, pooled
                     control FPR <= target at every threshold.
  (B) same as v4     on the four standard backgrounds, v5's thresholds equal detection v4's on
                     the v7 control runs (v8 defaults == v7, same seeds).
  (C) runner         two shards + merge: one row per (run, method), a finished shard is skipped,
                     merged rows equal a direct run, wallets for the honest sweep only.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.dirname(_CAP)
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from detection.v4 import experiment as E4
from detection.v4.fusion import METHODS, TARGETS
from detection.v5 import experiment as E
from washtrade.v7.generate import merge as v7_merge, run_sweep as v7_run
from washtrade.v8.generate import dataset_root, merge as v8_merge, run_sweep as v8_run


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    with tempfile.TemporaryDirectory() as data:
        v8_run("honest", seeds=2, out_dir=data, verbose=False)
        v8_merge("honest", out_dir=data, verbose=False)
        v8_run("adversary", seeds=1, out_dir=data, limit=6, verbose=False)
        v8_run("adversary", seeds=1, out_dir=data, shard=1, n_shards=2, limit=6, verbose=False)
        v8_merge("adversary", out_dir=data, verbose=False)

        # ------------------------------------------------------------------ A
        print("\n===== A) null =====")
        control, th = E.build_null(data, jobs=1, verbose=False)
        bgs = sorted(control["background"].unique())
        check("A eight backgrounds in the null", len(bgs) == 8, str(bgs))
        worst = max(float((th[f"fpr_{t}"] - TARGETS[t]).max()) for t in TARGETS)
        check("A pooled control FPR <= target everywhere", worst <= 1e-12, f"max excess {worst:.2e}")

        # ------------------------------------------------------------------ B
        print("\n===== B) same thresholds as v4 on the standard backgrounds =====")
        with tempfile.TemporaryDirectory() as d7:
            v7_run("scenarios", seeds=2, out_dir=d7, verbose=False)
            v7_merge("scenarios", out_dir=d7, verbose=False)
            c4, th4 = E4.build_null(d7, jobs=1, verbose=False)
        std = ["thin", "thick", "active", "active_dense"]
        a = th[th["background"].isin(std)].set_index(["background", "method", "floor"]).sort_index()
        b = th4[th4["background"].isin(std)].set_index(["background", "method", "floor"]).sort_index()
        cols = [c for c in a.columns if c.startswith(("theta_", "fpr_"))]
        check("B v5 thresholds == v4 thresholds (thin, thick, active, active_dense)",
              np.allclose(a[cols].to_numpy(float), b.loc[a.index, cols].to_numpy(float),
                          equal_nan=True))

        # ------------------------------------------------------------------ C
        print("\n===== C) runner =====")
        n0 = E.run_shard("adversary", 0, 2, data=data, verbose=False)
        n1 = E.run_shard("adversary", 1, 2, data=data, verbose=False)
        again = E.run_shard("adversary", 0, 2, data=data, verbose=False)
        m = E.merge("adversary", data=data, verbose=False)["results"]
        n = len(pd.read_csv(os.path.join(dataset_root("adversary", out_dir=data), "manifest.csv")))
        check("C two shards cover the sweep", n0 + n1 == n, f"{n0} + {n1} of {n}")
        check("C a finished shard is skipped", again == 0)
        check("C one row per (run, method)", len(m) == n * len(METHODS)
              and m.groupby(["run_id", "method"]).size().max() == 1)
        check("C v8 design factors carried", {"f_pipeline", "f_edge_order", "f_lam_wash"} <= set(m.columns))
        null, thr = E.load_null(data)
        man = pd.read_csv(os.path.join(dataset_root("adversary", out_dir=data), "manifest.csv"))
        r0 = man.iloc[-1]
        rows, _ = E.detect_one(os.path.join(dataset_root("adversary", out_dir=data), r0["path"]),
                               r0["background"], null, thr)
        direct = pd.DataFrame(rows).set_index("method")
        merged = m[m["run_id"] == r0["run_id"]].set_index("method")[direct.columns]
        check("C merged rows == a direct run", np.allclose(direct.to_numpy(float),
                                                           merged.to_numpy(float), equal_nan=True))
        E.run_shard("honest", 0, 1, data=data, verbose=False)
        mh = E.merge("honest", data=data, verbose=False)
        check("C wallets for honest, not for adversary",
              len(mh["wallets"]) > 0 and not os.path.exists(os.path.join(E.out_dir("adversary", data),
                                                                          "wallets.csv")))
        ctrl = mh["results"][mh["results"]["f_scenario"] == "control"]
        pooled = ctrl.groupby(["background", "method"])[["trade_fp", "trade_neg"]].sum()
        check("C control runs re-scored in pass 2: pooled FPR <= 1%",
              bool((pooled["trade_fp"] / pooled["trade_neg"] <= 0.01 + 1e-12).all()))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
