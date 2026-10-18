r"""run_smoke.py -- detection v4 smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root (generates its own small v7 datasets in a temporary folder):
  python "capstone codes/detection/v4/run_smoke.py"

  (A) closure = v1     the closure channel's wallet scores and pair scores equal v1's detector.
  (B) channels         round trips and same-pair repeats on hand-made tapes give the expected shares.
  (C) network step     every propagated channel keeps the volume-weighted mean (Proposition 2),
                       and the _noprop ablations equal x0.
  (D) fusion           tail probabilities and thresholds on synthetic data; fusing one channel ranks
                       pairs like its raw score; at every theta the pooled control FPR is <= target,
                       and a looser target never raises theta.
  (E) runner           null + two shards + merge: one row per (run, method), a finished shard is
                       skipped, merged rows equal a direct run, wallets only for scenarios.
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

from detection.v1.pipeline import detect
from detection.v4 import experiment as E
from detection.v4.channels import repeat_flags, repeat_scores, roundtrip_share
from detection.v4.detector import CHANNELS, NOPROP, detect_v4
from detection.v4.fusion import (FLOORS, FUSIONS, MAIN_FLOOR, METHODS, SINGLE, TARGETS, Null,
                                 Survival, floor_tag, theta_for)
from washtrade.v7.export import load_observable
from washtrade.v7.generate import dataset_root, merge as v7_merge, run_sweep


def _tape(rows):
    """[(time, seller, buyer, qty)] -> observable trades (one sell leg + one buy leg per row)."""
    legs, k = [], 0
    for t, s, b, q in rows:
        legs.append(dict(leg_id=k, time=t, wallet=s, side="SELL", qty=q, price=100.0))
        legs.append(dict(leg_id=k + 1, time=t, wallet=b, side="BUY", qty=q, price=100.0))
        k += 2
    return pd.DataFrame(legs)


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ B
    print("\n===== B) channels on hand-made tapes =====")
    check("B buy then sell 10 steps later, W=50 -> 1", roundtrip_share([0, 10], [1, -1], 50) == 1.0)
    check("B ... W=5 -> 0", roundtrip_share([0, 10], [1, -1], 5) == 0.0)
    check("B +2 then -1 -> 2 of 3 units matched", np.isclose(roundtrip_share([0, 5], [2, -1], 50), 2 / 3))
    check("B FIFO: +1, +1 (late), -1 matches the oldest still in window",
          roundtrip_share([0, 40, 60], [1, 1, -1], 50) == 2 / 3)
    check("B one-sided trading -> 0", roundtrip_share([0, 1, 2], [1, 1, 1], 50) == 0.0)
    pairs = pd.DataFrame(dict(time=[0, 10, 100, 5, 7, 7], seller=["a", "a", "a", "b", "c", "d"],
                              buyer=["b", "b", "b", "a", "d", "d"], qty=[1.0] * 6))
    f = repeat_flags(pairs, 20)
    check("B repeats: a->b at 0,10 repeat; a->b at 100 not; b->a not (direction); self never",
          f.tolist() == [True, True, False, False, False, False], str(f.tolist()))
    rs = repeat_scores(pairs, (20,))["repeat_20"]
    check("B repeat share: a 2/4, b 2/4, c 0", np.isclose(rs["a"], 0.5) and np.isclose(rs["b"], 0.5)
          and rs["c"] == 0.0, rs.round(3).to_dict().__repr__())
    # a fixed triad, one unit per event, every 3 steps: every channel sees it fully
    tri = _tape([(3 * i, "abc"[i % 3], "abc"[(i + 1) % 3], 1) for i in range(30)])
    det = detect_v4(tri)
    x0 = det.wallets[[f"x0_{c}" for c in CHANNELS]]
    check("B fixed triad: closure, round-trip and repeat (W >= 9) x0 = 1",
          bool((x0.drop(columns=["x0_repeat_20"]) == 1).all().all()), x0.mean().round(2).to_dict().__repr__())

    with tempfile.TemporaryDirectory() as data:
        run_sweep("scenarios", seeds=2, out_dir=data, verbose=False)
        v7_merge("scenarios", out_dir=data, verbose=False)
        run_sweep("rotation", seeds=1, out_dir=data, limit=12, verbose=False)
        v7_merge("rotation", out_dir=data, verbose=False)
        sroot = dataset_root("scenarios", out_dir=data)
        sman = pd.read_csv(os.path.join(sroot, "manifest.csv"))

        # ------------------------------------------------------------------ A
        print("\n===== A) closure channel = v1 =====")
        good = True
        for r in sman.iloc[::7].itertuples():
            tr = load_observable(os.path.join(sroot, r.path))["trades"]
            a, b = detect_v4(tr), detect(tr)
            good &= np.allclose(a.wallets["x_closure"], b.wallets["x"]) and np.allclose(
                a.wallets["x0_closure"], b.wallets["x0"]) and np.allclose(
                a.pairs["closure"].fillna(-1), b.pairs["score"].fillna(-1))
        check("A x0, x and pair scores equal v1 on every 7th scenarios run", good)

        # ------------------------------------------------------------------ C
        print("\n===== C) network step =====")
        good_c, good_n = True, True
        for r in sman.iloc[::5].itertuples():
            det = detect_v4(load_observable(os.path.join(sroot, r.path))["trades"])
            v = det.wallets["volume"].to_numpy()
            for ch in CHANNELS:
                good_c &= np.isclose(v @ det.wallets[f"x_{ch}"], v @ det.wallets[f"x0_{ch}"], rtol=1e-4)
            for ch in NOPROP:
                p = det.pairs
                x0 = det.wallets[f"x0_{ch}"]
                want = np.where(p["in_graph"], np.minimum(p["buyer"].map(x0), p["seller"].map(x0)), np.nan)
                good_n &= np.allclose(np.nan_to_num(want, nan=-1), p[f"{ch}_noprop"].fillna(-1))
        check("C volume-weighted mean of x preserved for every channel", good_c)
        check("C _noprop pair score = min of the two x0", good_n)

        # ------------------------------------------------------------------ D
        print("\n===== D) fusion and thresholds =====")
        sv = Survival([0.1, 0.5, 0.5, 0.9, np.nan], [1, 1, 1, 1, 1], total=5)
        check("D survival: tail shares incl. ties, above-all = 0, NaN stays NaN",
              np.allclose(sv(np.array([0.0, 0.5, 0.6, 0.95])), [0.8, 0.6, 0.2, 0.0])
              and np.isnan(sv(np.array([np.nan]))[0]))
        check("D theta: smallest with FPR <= target", theta_for([0.9, 0.5, 0.5, 0.1], [1] * 4, 100, 0.01) == 0.9
              and theta_for([0.9, 0.5, 0.5, 0.1], [1] * 4, 100, 0.03) == 0.5)
        check("D theta: unreachable -> just above the top score",
              theta_for([0.9, 0.9], [5, 5], 100, 0.01) > 0.9)

        control, th = E.build_null(data, jobs=1, verbose=False)
        null = Null(control)
        good = True
        for bg, d in control.groupby("background"):
            z = null.method_scores(bg, d)
            for ch in CHANNELS:
                raw = d[ch].to_numpy()
                single = 1.0 - null.maps[bg][ch](raw)
                okm = ~np.isnan(raw)
                o = np.argsort(raw[okm], kind="mergesort")
                good &= bool(np.all(np.diff(single[okm][o]) >= -1e-12))
        check("D one-channel z is monotone in its raw score", good)
        fpr_cols = [f"fpr_{t}" for t in TARGETS]
        worst = max(float((th[c] - TARGETS[c[4:]]).max()) for c in fpr_cols)
        check("D pooled control FPR at theta <= target, every (background, method, floor, target)",
              worst <= 1e-12 and th["fpr_1"].notna().all(), f"max excess {worst:.2e}")
        mf = th[th["floor"] == MAIN_FLOOR]
        check("D looser target never raises theta (main floor)",
              bool((mf["theta_5"] <= mf["theta_1"]).all() and (mf["theta_1"] <= mf["theta_0.1"]).all()))
        check("D thresholds for 5 backgrounds x every method x every floor",
              len(th) == 5 * len(METHODS) * len(FLOORS) and set(th["method"]) == set(METHODS))
        good_f = True
        for bg, d in control.groupby("background"):
            for fl in FLOORS:
                sc = null.method_scores(bg, d, fl)
                for m in SINGLE:
                    raw = d[m].to_numpy()
                    good_f &= bool(np.all(sc[m].to_numpy()[raw < fl] < 0))
                below = np.all(np.vstack([d[c].to_numpy() < fl for c in FUSIONS["v4"]]), axis=0)
                good_f &= bool(np.all(sc["v4"].to_numpy()[below] == 0))
            z0 = null.method_scores(bg, d, 0.0)
            good_f &= bool(np.allclose(z0[SINGLE].to_numpy(), d[SINGLE].to_numpy(), equal_nan=True))
        check("D floor: below-floor scores never count (single -> -1, fused -> 0); floor 0 = raw", good_f)
        check("D fusions cover the channels", set(FUSIONS["v4"]) == set(CHANNELS))

        # ------------------------------------------------------------------ E
        print("\n===== E) runner =====")
        n0 = E.run_shard("rotation", 0, 2, data=data, verbose=False)
        n1 = E.run_shard("rotation", 1, 2, data=data, verbose=False)
        again = E.run_shard("rotation", 0, 2, data=data, verbose=False)
        m = E.merge("rotation", data=data, verbose=False)["results"]
        check("E two shards cover the sweep", n0 + n1 == 12, f"{n0} + {n1}")
        check("E a finished shard is skipped", again == 0)
        check("E one row per (run, method)", len(m) == 12 * len(METHODS)
              and m.groupby(["run_id", "method"]).size().max() == 1)
        check("E v7 design factors carried", {"f_pool", "f_cycle_len", "f_gap"} <= set(m.columns))
        rroot = dataset_root("rotation", out_dir=data)
        r0 = pd.read_csv(os.path.join(rroot, "manifest.csv")).iloc[3]
        rows, _ = E.detect_one(os.path.join(rroot, r0["path"]), r0["background"], null, th)
        direct = pd.DataFrame(rows).set_index("method")
        merged = m[m["run_id"] == r0["run_id"]].set_index("method")[direct.columns]
        check("E merged rows == a direct run", np.allclose(direct.to_numpy(float), merged.to_numpy(float),
                                                           equal_nan=True))
        E.run_shard("scenarios", 0, 1, data=data, verbose=False)
        ms = E.merge("scenarios", data=data, verbose=False)
        check("E wallets table for scenarios, not for rotation",
              len(ms["wallets"]) > 0 and not os.path.exists(os.path.join(E.out_dir("rotation", data), "wallets.csv")))
        ctrl = ms["results"][ms["results"]["f_scenario"] == "control"]
        fp_cols = ["trade_fp"] + [f"trade_fp_{floor_tag(f)}" for f in FLOORS if f != MAIN_FLOOR]
        pooled = ctrl.groupby(["background", "method"])[fp_cols + ["trade_neg"]].sum()
        check("E control runs re-scored in pass 2: pooled FPR <= 1% at every floor (same runs as the null)",
              bool((pooled[fp_cols].div(pooled["trade_neg"], axis=0) <= 0.01 + 1e-12).all().all()))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
