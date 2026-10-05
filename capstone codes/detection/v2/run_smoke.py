r"""run_smoke.py -- detection v2 smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root (needs the v6 scenarios dataset for A-D; E generates its own data):
  python "capstone codes/detection/v2/run_smoke.py"

  (A) v1 parity        on v6 runs, every v1 metric from v2's evaluate equals v1's evaluate_run.
  (B) truth labels     same-group wash volume <= wash volume; equal on single-group runs.
  (C) group table      leakage matches v6's own pro-rata leakage on runs without self-pairs;
                       the exact identity x_bar = x0_bar + l * (s_bar - x_out) holds at the fixed
                       point (x_out = outside-volume-weighted score of the group's wallets).
  (D) curves           at theta = 0.9 the curve reproduces Algorithm 1's trade recall / FPR.
  (E) shards + merge   two shards of a small generated sweep merge into one row per run/config.
  (F) random walk      the closed-form return probability matches a Monte Carlo walk.
  (G) wallet table     one row per graph wallet; closed <=> x0 = 1 (single market); truth joins.
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

from detection.v1.evaluate import evaluate_run as v1_evaluate_run
from detection.v1.pipeline import DetectorConfig, detect
from detection.v2.closing import p_return, wallet_table
from detection.v2.evaluate import attach_truth, evaluate, group_table, threshold_curve
from detection.v2.experiment import merge, run_shard
from washtrade.v6.export import load_labels, load_meta, load_observable
from washtrade.v6.generate import DEFAULT_OUT, dataset_root, merge as v6_merge, run_sweep

RUNS = ["thin__triad__s000", "thick__triad_gap20__s001", "active__hard_hi__s000",
        "active__triad_camo_hi__s002", "thin__two_groups__s003", "active_dense__hard__s004"]


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    base = os.path.join(dataset_root("scenarios", out_dir=DEFAULT_OUT), "runs")
    if not os.path.isdir(base):
        print("v6 scenarios dataset not found: run washtrade/v6 generate.py run --sweep scenarios")
        return 1

    for rid in RUNS:
        d = os.path.join(base, rid)
        trades, labels, meta = load_observable(d)["trades"], load_labels(d), load_meta(d)
        det = detect(trades, DetectorConfig(), horizon=meta["config"]["sim_time"])
        print(f"\n===== {rid} =====")

        e1, e2 = v1_evaluate_run(d), evaluate(det, labels)
        check("A every v1 metric identical", all(_same(e1[k], e2[k]) for k in e1))

        pairs, _ = attach_truth(det, labels)
        wash = pairs.loc[pairs["is_wash"], "qty"].sum()
        same = pairs.loc[pairs["is_wash_same_group"], "qty"].sum()
        n_groups = len(meta["groups"])
        check("B same-group wash volume <= wash volume (equal with one group)",
              same <= wash + 1e-9 and (n_groups > 1 or abs(same - wash) < 1e-9),
              f"{same:.1f} / {wash:.1f}")

        g = group_table(det, labels)
        if det.stats["self_pair_qty"] == 0:
            v6 = [grp["leak_prorata"] for grp in meta["groups"]]
            check("C leakage equals v6's pro-rata leakage (no self-pairs)",
                  np.allclose(g["leak"].to_numpy(), v6, atol=1e-9),
                  f"{np.round(g['leak'].to_numpy(), 3).tolist()} vs {np.round(v6, 3).tolist()}")
        pos = {w: i for i, w in enumerate(det.index)}
        x = det.wallets["x"].to_numpy()
        V = det.V
        wl = labels["wallets"]
        errs = []
        for k, row in g.iterrows():
            members = wl.loc[wl["group_id"] == row["group_id"], "wallet"]
            idx = np.array([pos[w] for w in members if w in pos])
            others = np.setdiff1d(np.arange(len(x)), idx)
            out_j = V[np.ix_(idx, others)].sum(axis=1)
            x_out = (out_j * x[idx]).sum() / out_j.sum() if out_j.sum() else 0.0
            s_bar = row["s_bar"] if np.isfinite(row["s_bar"]) else 0.0
            errs.append(abs(row["x_bar"] - (row["x0_bar"] + row["leak"] * (s_bar - x_out))))
        check("C exact identity x_bar = x0_bar + l (s_bar - x_out) at the fixed point",
              max(errs) < 1e-3, f"max error {max(errs):.1e}")

        c = threshold_curve(det, labels)
        r = c[np.isclose(c["theta"], 0.9)].iloc[0]
        rec = r.trade_tp / r.trade_pos if r.trade_pos else float("nan")
        fpr = r.trade_fp / r.trade_neg if r.trade_neg else float("nan")
        w = wallet_table(det, trades, labels)
        n_wash = int(labels["wallets"].set_index("wallet")["is_wash"].reindex(det.index).fillna(False).sum())
        check("G wallet table: one row per graph wallet, closed <=> x0 = 1, wash wallets match",
              len(w) == len(det.index) and bool((w["closed"] == (w["x0"] == 1.0)).all())
              and int(w["is_wash"].sum()) == n_wash, f"{len(w)} wallets, {n_wash} wash")

        check("D curve at 0.9 = Algorithm 1 recall and FPR",
              _same(round(rec, 12), round(e2["trade_fixed_recall"], 12))
              and _same(round(fpr, 12), round(e2["trade_fixed_fpr"], 12)),
              f"recall {rec:.3f}, FPR {fpr:.3f}")

    print("\n===== E) shards + merge =====")
    with tempfile.TemporaryDirectory() as tmp:
        run_sweep("price", seeds=1, out_dir=tmp, limit=4, verbose=False)
        v6_merge("price", out_dir=tmp, verbose=False)
        n0 = run_shard("price", 0, 2, data=tmp, verbose=False)
        n1 = run_shard("price", 1, 2, data=tmp, verbose=False)
        again = run_shard("price", 0, 2, data=tmp, verbose=False)
        m = merge("price", data=tmp, verbose=False)
        res = m["results"]
        check("E two shards cover every run once per config",
              n0 + n1 == 4 and len(res) == 4 * res["config"].nunique()
              and res.groupby("config")["run_id"].nunique().eq(4).all(), f"{n0}+{n1} runs")
        check("E a finished shard is skipped", again == 0)
        check("E curves, groups and wallets merged",
              len(m["curves"]) > 0 and len(m["groups"]) > 0 and len(m["wallets"]) > 0)

    print("\n===== F) random walk =====")
    rng = np.random.default_rng(0)
    check("F closed form: p(0) = 0, p(2) = 0.5, p(3) = 0.5",
          p_return(0) == 0.0 and p_return(2) == 0.5 and p_return(3) == 0.5)
    for n in (6, 20):
        walks = np.cumsum(rng.choice([-1, 1], size=(200_000, n)), axis=1)
        mc = float((walks == 0).any(axis=1).mean())
        check(f"F n = {n}: closed form {p_return(n):.4f} vs Monte Carlo {mc:.4f}",
              abs(mc - p_return(n)) < 0.005)

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
