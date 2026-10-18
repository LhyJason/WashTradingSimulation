r"""run_smoke.py -- detection v3 smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root (generates its own small v7 datasets in a temporary folder):
  python "capstone codes/detection/v3/run_smoke.py"

  (A) shards + merge   two shards of a small v7 sweep merge into one row per (run, config), with
                       the v7 design factors; a finished shard is skipped on re-run.
  (B) same detector    v3's rows equal v2's evaluate on the same run (v3 adds no detector).
  (C) pool groups      the group table counts the whole pool (n_wallets) and, in the graph, exactly
                       the pool wallets that traded.
  (D) calibration      synthetic curves give the expected threshold (and inf when unreachable);
                       on real control runs the pooled FPR at theta_cal is <= target and theta_cal
                       does not rise when the target is relaxed; every run gets one row per config.
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

from detection.v1.pipeline import DetectorConfig, detect
from detection.v2.evaluate import evaluate, group_table
from detection.v3 import calibrate as C
from detection.v3.experiment import merge, out_dir, run_shard
from washtrade.v7.export import load_labels, load_meta, load_observable
from washtrade.v7.generate import dataset_root, merge as v7_merge, run_sweep


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    with tempfile.TemporaryDirectory() as data:
        # small datasets: scenarios at 2 seeds (has the control runs), rotation's first 12 runs
        run_sweep("scenarios", seeds=2, out_dir=data, verbose=False)
        v7_merge("scenarios", out_dir=data, verbose=False)
        run_sweep("rotation", seeds=1, out_dir=data, limit=12, verbose=False)
        v7_merge("rotation", out_dir=data, verbose=False)

        # ------------------------------------------------------------------ A
        print("\n===== A) shards + merge =====")
        n0 = run_shard("rotation", 0, 2, data=data, verbose=False)
        n1 = run_shard("rotation", 1, 2, data=data, verbose=False)
        again = run_shard("rotation", 0, 2, data=data, verbose=False)
        m = merge("rotation", data=data, verbose=False)
        res = m["results"]
        check("A two shards cover the sweep", n0 + n1 == 12, f"{n0} + {n1}")
        check("A a finished shard is skipped", again == 0)
        check("A one row per (run, config)", len(res) == 12 * 3
              and res.groupby(["run_id", "config"]).size().max() == 1)
        check("A v7 design factors carried", {"f_pool", "f_cycle_len", "f_gap"} <= set(res.columns))
        for sw in ("scenarios",):
            run_shard(sw, 0, 1, data=data, verbose=False)
            merge(sw, data=data, verbose=False)

        # ------------------------------------------------------------------ B
        print("\n===== B) same detector as v2 =====")
        root = dataset_root("rotation", out_dir=data)
        rid = res["run_id"].iloc[0]
        rdir = os.path.join(root, "runs", rid)
        det = detect(load_observable(rdir)["trades"], DetectorConfig(),
                     horizon=load_meta(rdir)["config"]["sim_time"])
        direct = evaluate(det, load_labels(rdir))
        row = res[(res["run_id"] == rid) & (res["config"] == "paper")].iloc[0]
        same = all((np.isnan(v) and np.isnan(row[k])) if isinstance(v, float) and np.isnan(v)
                   else np.isclose(v, row[k]) for k, v in direct.items()
                   if isinstance(v, (int, float, np.floating)))
        check("B v3 row == v2 evaluate on the same run", same, rid)

        # ------------------------------------------------------------------ C
        print("\n===== C) pool groups =====")
        man = pd.read_csv(os.path.join(root, "manifest.csv"))
        good = True
        for r in man.itertuples():
            rd = os.path.join(root, r.path)
            lab = load_labels(rd)
            det = detect(load_observable(rd)["trades"], DetectorConfig(),
                         horizon=load_meta(rd)["config"]["sim_time"])
            g = group_table(det, lab).iloc[0]
            w = lab["wallets"]
            traded = int(((w["is_wash"].astype(bool)) & (w["n_legs"] > 0)).sum())
            good &= int(g["n_wallets"]) == int(r.f_pool) and int(g["n_in_graph"]) == traded
        check("C group table: whole pool counted, traded pool wallets in the graph", good,
              f"pools {sorted(man['f_pool'].unique())}")

        # ------------------------------------------------------------------ D
        print("\n===== D) calibration =====")
        syn = pd.DataFrame([dict(background="b", config="paper", run_id="r", theta=t,
                                 trade_fp=fp, trade_neg=100.0)
                            for t, fp in ((0.8, 5.0), (0.9, 2.0), (0.95, 0.5), (1.0, 0.0))])
        th = C.control_thresholds(syn, target=0.01)
        check("D synthetic: smallest theta with FPR <= 1%", th["theta_cal"].iloc[0] == 0.95)
        th0 = C.control_thresholds(syn.assign(trade_fp=50.0), target=0.01)
        check("D synthetic: unreachable target -> inf", np.isinf(th0["theta_cal"].iloc[0]))

        thr = C.calibrate_all(data, verbose=False)
        check("D thresholds for 5 backgrounds x 2 curve configs", len(thr) == 10
              and set(thr["config"]) == {"paper", "no_propagation"})
        check("D pooled control FPR at theta_cal <= target", bool((thr["fpr_at_cal"] <= 0.01).all()),
              thr.loc[thr["config"] == "paper", ["background", "theta_cal"]].to_string(index=False)
              .replace("\n", "; "))
        loose = C.calibrate_all(data, target=0.05, verbose=False)
        mrg = thr.merge(loose, on=["background", "config"], suffixes=("", "_loose"))
        check("D relaxing the target never raises theta_cal",
              bool((mrg["theta_cal_loose"] <= mrg["theta_cal"]).all()))
        thr = C.calibrate_all(data, verbose=False)
        for sw, n in (("scenarios", 120), ("rotation", 12)):
            cal = pd.read_csv(os.path.join(out_dir(sw, data), "calibrated.csv"))
            check(f"D {sw}: one calibrated row per (run, curve config)",
                  len(cal) == 2 * n and cal.groupby(["run_id", "config"]).size().max() == 1,
                  f"{len(cal)} rows")
        cal = pd.read_csv(os.path.join(out_dir("scenarios", data), "calibrated.csv"))
        ctrl = cal[cal["f_scenario"] == "control"]
        check("D control runs have no wash volume (recall undefined)",
              bool(ctrl["trade_pos"].eq(0).all()))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
