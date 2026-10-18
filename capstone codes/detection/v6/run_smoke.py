r"""run_smoke.py -- detection v6 smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root (generates its own small v8 dataset in a temporary folder):
  python "capstone codes/detection/v6/run_smoke.py"

  (A) cluster channel  on hand-made graphs: two dense groups are found; the wallet score matches
                       (a - e) / max(1 - e, a); a leaf wallet scores high but its pair with an
                       honest wallet does not; pairs across communities score 0, self-pairs NaN;
                       one community covering every wallet scores 0; a closed community holding
                       most of the volume is scored by its lift, below the floor; Louvain is
                       deterministic for a fixed seed.
  (B) v4 unchanged     detect_v6's v4 columns equal detect_v4's; v6's thresholds for v4's methods
                       equal detection v5's on the same control runs.
  (C) null             pooled control FPR <= target for every method, cluster ones included; the
                       applicability table covers every background; where the channel is off,
                       cluster methods flag nothing and v6's scores equal v4's.
  (D) runner           two shards + merge: one row per (run, method), a finished shard is skipped,
                       merged rows equal the parts and a direct run, wallets for honest only.
  (E) owner label      touch + far adds up; control runs have no same-owner volume.
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

from detection.v4.detector import SCORES as V4_SCORES
from detection.v4.detector import detect_v4
from detection.v4.fusion import METHODS as V4_METHODS
from detection.v4.fusion import TARGETS
from detection.v5 import experiment as E5
from detection.v6 import cluster as C
from detection.v6 import experiment as E
from detection.v6.detector import detect_v6
from detection.v6.fusion import METHODS
from washtrade.v8.export import load_observable
from washtrade.v8.generate import dataset_root, merge as v8_merge, run_sweep as v8_run


def graph(edges, n):
    V = np.zeros((n, n))
    for i, j, w in edges:
        V[i, j] += w
        V[j, i] += w
    return V


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) cluster channel on hand-made graphs =====")
    # two triangles with heavy internal volume, weak links between them, and a leaf on wallet 3
    E_ = [(0, 1, 10), (1, 2, 10), (0, 2, 10), (3, 4, 10), (4, 5, 10), (3, 5, 10),
          (0, 3, 1), (2, 4, 1), (3, 6, 1)]
    V = graph(E_, 7)
    lab = C.communities(V, 1.0, 0)
    check("A two groups found, leaf with its neighbour",
          len({lab[0], lab[1], lab[2]}) == 1 and len({lab[3], lab[4], lab[5], lab[6]}) == 1
          and lab[0] != lab[3])
    s = C.wallet_scores(V, lab)
    vol = V.sum(axis=1)
    idx = np.where(lab == lab[0])[0]
    a0 = V[0, idx].sum() / vol[0]
    e0 = (vol[idx].sum() - vol[0]) / (vol.sum() - vol[0])
    check("A wallet score = (a - e) / max(1 - e, a)", np.isclose(s[0], (a0 - e0) / max(1 - e0, a0)),
          f"s_0 = {s[0]:.3f}")
    idx6 = np.where(lab == lab[6])[0]
    e6 = (vol[idx6].sum() - vol[6]) / (vol.sum() - vol[6])
    check("A leaf wallet (a = 1) scores 1 - e", np.isclose(s[6], 1 - e6), f"s_6 = {s[6]:.3f}")
    bi = np.array([0, 3, 6, 0, 2])
    si = np.array([1, 6, 3, 3, 2])
    ing = np.array([True, True, True, True, False])
    ps = C.pair_scores(lab, s, bi, si, ing)
    check("A pair = min of its wallets (leaf pair = the honest side)",
          np.isclose(ps[0], min(s[0], s[1])) and np.isclose(ps[1], min(s[3], s[6]))
          and ps[1] < 1 and np.isclose(ps[1], ps[2]))
    check("A across communities 0, self-pair NaN", ps[3] == 0.0 and np.isnan(ps[4]))
    one = np.zeros(7, dtype=int)
    check("A one community over every wallet scores 0", np.allclose(C.wallet_scores(V, one), 0))
    # a closed block holding ~80% of the volume next to a closed ~20% block: lift caps the big one
    big = graph([(0, 1, 40), (1, 2, 40), (0, 2, 40), (3, 4, 10), (4, 5, 10), (3, 5, 10), (2, 3, 1)], 6)
    lb = np.array([0, 0, 0, 1, 1, 1])
    sb = C.wallet_scores(big, lb)
    vb = big.sum(axis=1)
    ab = big[0, :3].sum() / vb[0]
    eb = (vb[:3].sum() - vb[0]) / (vb.sum() - vb[0])
    check("A majority closed block scored by its lift 1 - e/a (< 0.5 here)",
          np.isclose(sb[0], 1 - eb / ab) and sb[0] < 0.5 and (ab - eb) / (1 - eb) > 0.9,
          f"lift score {sb[0]:.2f}, excess {(ab - eb) / (1 - eb):.2f}")
    rng = np.random.default_rng(1)
    R = np.triu(rng.integers(0, 3, (30, 30)) * (rng.random((30, 30)) < 0.2), 1).astype(float)
    R = R + R.T
    check("A Louvain deterministic for a fixed seed",
          np.array_equal(C.communities(R, 1.0, 0), C.communities(R, 1.0, 0)))

    with tempfile.TemporaryDirectory() as data:
        v8_run("honest", seeds=2, out_dir=data, verbose=False)
        v8_merge("honest", out_dir=data, verbose=False)
        v8_run("adversary", seeds=1, out_dir=data, limit=6, verbose=False)
        v8_run("adversary", seeds=1, out_dir=data, shard=1, n_shards=2, limit=6, verbose=False)
        v8_merge("adversary", out_dir=data, verbose=False)

        # ------------------------------------------------------------------ B
        print("\n===== B) v4 unchanged =====")
        root = dataset_root("adversary", out_dir=data)
        man = pd.read_csv(os.path.join(root, "manifest.csv"))
        good = True
        for r in man.itertuples():
            tr = load_observable(os.path.join(root, r.path))["trades"]
            a, b = detect_v6(tr).pairs, detect_v4(tr).pairs
            good &= np.allclose(a[V4_SCORES].to_numpy(float), b[V4_SCORES].to_numpy(float), equal_nan=True)
        check("B detect_v6's v4 columns == detect_v4", good, f"{len(man)} runs")
        control, th = E.build_null(data, jobs=1, verbose=False)
        _, th5 = E5.build_null(data, jobs=1, verbose=False)
        a = th[th["method"].isin(V4_METHODS)].set_index(["background", "method", "floor"]).sort_index()
        b = th5.set_index(["background", "method", "floor"]).sort_index()
        cols = [c for c in b.columns if c.startswith(("theta_", "fpr_"))]
        check("B thresholds for v4's methods == detection v5's",
              a.index.equals(b.index) and np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float),
                                                      equal_nan=True))

        # ------------------------------------------------------------------ C
        print("\n===== C) null =====")
        check("C every method has thresholds", set(th["method"]) == set(METHODS))
        worst = max(float((th[f"fpr_{t}"] - TARGETS[t]).max()) for t in TARGETS)
        check("C pooled control FPR <= target everywhere", worst <= 1e-12, f"max excess {worst:.2e}")
        check("C cluster scores in [0, 1] (NaN = self-pair)",
              bool(control[["cluster", "cluster_r2"]].dropna().stack().between(0, 1).all()))
        app = pd.read_csv(E.backgrounds_path(data))
        check("C applicability for every background", set(app["background"]) == set(control["background"]),
              "; ".join(f"{b} {m:.2f} {'on' if o else 'off'}" for b, m, o in
                        zip(app["background"], app["control_modularity"], app["cluster_enabled"])))
        null0, _ = E.load_null(data)
        good = True
        for bg in app["background"]:
            null0.enabled[bg] = False
            d = control[control["background"] == bg]
            sc = null0.method_scores(bg, d)
            good &= bool(np.array_equal(sc["v6"].to_numpy(), sc["v4"].to_numpy(), equal_nan=True))
            good &= bool((sc["cluster"].dropna() == -1).all())
        check("C channel off: v6 == v4 and cluster never above the floor", good)

        # ------------------------------------------------------------------ D
        print("\n===== D) runner =====")
        n0 = E.run_shard("adversary", 0, 2, data=data, verbose=False)
        n1 = E.run_shard("adversary", 1, 2, data=data, verbose=False)
        again = E.run_shard("adversary", 0, 2, data=data, verbose=False)
        m = E.merge("adversary", data=data, verbose=False)["results"]
        n = len(man)
        check("D two shards cover the sweep", n0 + n1 == n, f"{n0} + {n1} of {n}")
        check("D a finished shard is skipped", again == 0)
        check("D one row per (run, method)", len(m) == n * len(METHODS)
              and m.groupby(["run_id", "method"]).size().max() == 1)
        parts = pd.concat([pd.read_csv(os.path.join(E.out_dir("adversary", data), "parts",
                                                    f"results_{i:03d}of002.csv"),
                                       float_precision="round_trip") for i in range(2)],
                          ignore_index=True)
        mm = pd.read_csv(os.path.join(E.out_dir("adversary", data), "results.csv"),
                         float_precision="round_trip")
        num = mm.select_dtypes("number").columns
        check("D merged file == parts (round-trip parsing)",
              np.array_equal(mm[num].to_numpy(float), parts[num].to_numpy(float), equal_nan=True))
        null, thr = E.load_null(data)
        r0 = man.iloc[-1]
        rows, _ = E.detect_one(os.path.join(root, r0["path"]), r0["background"], null, thr)
        direct = pd.DataFrame(rows).set_index("method")
        merged = m[m["run_id"] == r0["run_id"]].set_index("method")[direct.columns]
        check("D merged rows == a direct run", np.allclose(direct.to_numpy(float),
                                                           merged.to_numpy(float), equal_nan=True))
        check("D community statistics carried",
              {"n_comm_cluster", "modularity_cluster", "n_comm_cluster_r2"} <= set(m.columns))
        E.run_shard("honest", 0, 1, data=data, verbose=False)
        mh = E.merge("honest", data=data, verbose=False)
        check("D wallets for honest (with communities), not for adversary",
              len(mh["wallets"]) > 0 and "comm_cluster" in mh["wallets"].columns
              and not os.path.exists(os.path.join(E.out_dir("adversary", data), "wallets.csv")))
        ctrl = mh["results"][mh["results"]["f_scenario"] == "control"]
        pooled = ctrl.groupby(["background", "method"])[["trade_fp", "trade_neg"]].sum()
        check("D control runs re-scored in pass 2: pooled FPR <= 1%",
              bool((pooled["trade_fp"] / pooled["trade_neg"] <= 0.01 + 1e-12).all()))

        # ------------------------------------------------------------------ E
        print("\n===== E) owner label =====")
        allr = pd.concat([m, mh["results"]], ignore_index=True)
        check("E honest = touch + far, false positives = touch + far",
              np.allclose(allr["trade_neg_touch"] + allr["trade_neg_far"], allr["trade_neg_owner"])
              and np.allclose(allr["trade_fp_touch"] + allr["trade_fp_far"], allr["trade_fp_owner"]))
        check("E control runs: no same-owner volume, owner FPs == order FPs",
              (ctrl["trade_pos_owner"] == 0).all() and np.allclose(ctrl["trade_fp_owner"], ctrl["trade_fp"]))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
