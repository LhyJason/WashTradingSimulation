r"""run_smoke.py -- Part 2 detection v1 smoke test (headless).

Run (from the repo root):
  python "capstone codes/detection/v1/run_smoke.py"

  (A) closures       -> the paper's closure definition on hand-built position paths, incl. the
                        reversal split and Example 7 (Mazric: 30,128 bought, 30,125 sold back).
  (B) x0 windows     -> n_windows=1 is the literal 1{Q > 0}; K windows weight by volume share.
  (C) pairing        -> pro-rata split is exact on 1-vs-n clears, conserves every leg's quantity,
                        rejects unbalanced clears.
  (D) propagation    -> Proposition 1 (converges to the closed form) and Proposition 2 (v'x(k)
                        conserved) on a random symmetric graph.
  (E) threshold      -> Y(theta) and theta* on a hand-computed 4-wallet example; empty Theta.
  (F) toy detection  -> a wash triangle + honest traders + one interception: separated perfectly.
  (G) real v3 runs   -> end-to-end on generated data; the detector never opens labels/.
"""
import os
import shutil
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

from detection.v1 import (DetectorConfig, detect, detect_run, evaluate, evaluate_run,
                          find_closures, initial_scores, pair_trades, clear_stats, volume_graph,
                          propagate, row_normalize, stationary, select_threshold,
                          spillover_curve, r_scores)
from washtrade.v3.generate import generate


def tape(rows, price=100.0):
    """rows = [(time, wallet, side, qty), ...] -> observable trades frame."""
    df = pd.DataFrame(rows, columns=["time", "wallet", "side", "qty"])
    df["price"] = price
    df.insert(0, "leg_id", np.arange(len(df)))
    return df


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) closures =====")
    cases = [
        ("buy 1, sell 1", [1, -1], [1]),
        ("never back to zero", [1, 1, -1], []),
        ("dip is not terminal-closure, final exit is", [2, -1, 1, -2], [3]),
        ("reversal split: long 1 -> short 1", [1, -2], [1]),
        ("Example 7 (Mazric): 30128 then -30125", [30128, -30125], [1]),
        ("large residual is not a closure", [30128, -29000], []),
        ("repeated round trips", [1, -1, 1, -1], [1, 3]),
        ("staggered exit: only the last step is terminal", [3, -1, -1, -1], [3]),
        ("contraction to 0 right after a closure needs a new max", [1, -1, -1, 1], [1, 3]),
    ]
    for name, seq, want in cases:
        got = find_closures(seq)
        check(f"A {name}", got == want, f"{seq} -> {got}")

    # ------------------------------------------------------------------ B
    print("\n===== B) initial score windows =====")
    tr = tape([(10, "a", "BUY", 1), (20, "a", "SELL", 1),          # closes in window 0
               (60, "a", "BUY", 1), (70, "a", "BUY", 1),           # holds in window 1
               (10, "b", "SELL", 1), (20, "b", "BUY", 1),
               (60, "b", "SELL", 1), (70, "b", "SELL", 1)])
    x1 = initial_scores(tr, n_windows=1, horizon=100)
    x2 = initial_scores(tr, n_windows=2, horizon=100)
    check("B K=1 is the literal 1{Q>0}", x1.loc["a", "x0"] == 1.0, f"x0={x1.loc['a', 'x0']}")
    check("B K=2 weights by the volume share of windows with a closure",
          abs(x2.loc["a", "x0"] - 0.5) < 1e-12, f"x0={x2.loc['a', 'x0']}")
    same_clear = tape([(5, "c", "BUY", 1), (5, "c", "SELL", 1), (6, "c", "BUY", 1), (7, "c", "SELL", 1)])
    xs = initial_scores(same_clear)
    check("B a buy+sell in one clear is netted (no phantom closure), volume stays gross",
          xs.loc["c", "n_closures"] == 1 and xs.loc["c", "volume"] == 4.0,
          f"closures={xs.loc['c', 'n_closures']}, volume={xs.loc['c', 'volume']}")

    # ------------------------------------------------------------------ C
    print("\n===== C) pairing =====")
    tr = tape([(1, "a", "BUY", 1), (1, "b", "SELL", 1),                           # 1 x 1
               (2, "a", "BUY", 1), (2, "c", "BUY", 1), (2, "b", "SELL", 2),       # 2 x 1
               (3, "a", "BUY", 1), (3, "c", "BUY", 3), (3, "b", "SELL", 2), (3, "d", "SELL", 2)])  # 2 x 2
    p = pair_trades(tr)
    q = p.set_index(["time", "buyer", "seller"])["qty"]
    check("C 1x1 clear is exact", q.loc[(1, "a", "b")] == 1.0)
    check("C 1-seller clear gives each buyer its own quantity",
          q.loc[(2, "a", "b")] == 1.0 and q.loc[(2, "c", "b")] == 1.0)
    check("C 2x2 clear is split b*s/Q",
          np.allclose([q.loc[(3, "a", "b")], q.loc[(3, "a", "d")], q.loc[(3, "c", "b")], q.loc[(3, "c", "d")]],
                      [0.5, 0.5, 1.5, 1.5]))
    by_buy = p.groupby("buy_leg_id")["qty"].sum()
    by_sell = p.groupby("sell_leg_id")["qty"].sum()
    legs = tr.set_index("leg_id")["qty"]
    check("C every leg's quantity is conserved",
          np.allclose(by_buy, legs.reindex(by_buy.index)) and np.allclose(by_sell, legs.reindex(by_sell.index)))
    cs = clear_stats(tr)
    check("C clear_stats counts only clears with >1 wallet on BOTH sides as ambiguous",
          cs["n_ambiguous_clears"] == 1, str(cs))
    try:
        pair_trades(tape([(1, "a", "BUY", 2), (1, "b", "SELL", 1)]))
        raised = False
    except ValueError:
        raised = True
    check("C unbalanced clear is rejected", raised)
    w, V, self_q = volume_graph(pair_trades(tape([(1, "a", "BUY", 1), (1, "a", "SELL", 1),
                                                  (2, "a", "BUY", 1), (2, "b", "SELL", 1)])))
    check("C self pairs are dropped from V, V is symmetric", self_q == 1.0 and np.allclose(V, V.T)
          and V[w.index("a"), w.index("b")] == 1.0)

    # ------------------------------------------------------------------ D
    print("\n===== D) propagation =====")
    rng = np.random.default_rng(0)
    n = 12
    Vr = rng.integers(0, 4, size=(n, n)).astype(float) * (rng.random((n, n)) < 0.4)
    Vr = np.triu(Vr, 1)
    Vr = Vr + Vr.T
    Vr[0, 1] = Vr[1, 0] = 2.0                                   # make sure nobody is isolated
    Vr[np.arange(n), (np.arange(n) + 1) % n] += 1
    Vr[(np.arange(n) + 1) % n, np.arange(n)] += 1
    B = row_normalize(Vr)
    x0 = rng.random(n)
    x, hist, k = propagate(x0, B, tol=1e-12)
    xs = stationary(x0, B)
    check("D Proposition 1: iteration converges to (I - B/2) x = x0/2", np.max(np.abs(x - xs)) < 1e-9,
          f"{k} iterations, max|diff|={np.max(np.abs(x - xs)):.2e}")
    v = Vr.sum(axis=1)
    cons = hist @ v
    check("D Proposition 2: v'x(k) is conserved", np.max(np.abs(cons - cons[0])) / cons[0] < 1e-9,
          f"range {cons.min():.6f}..{cons.max():.6f}")
    _, _, k_default = propagate(x0, B)
    check("D default tol 1e-5 converges quickly", k_default < 50, f"{k_default} iterations")

    # ------------------------------------------------------------------ E
    print("\n===== E) threshold selection =====")
    #   A,B: wash pair (x .95); C: x .85, trades a little with A; D: x .2, trades with C
    xE = np.array([0.95, 0.95, 0.85, 0.20])
    VE = np.zeros((4, 4))
    for i, j, vol in ((0, 1, 10), (0, 2, 1), (2, 3, 5)):
        VE[i, j] = VE[j, i] = vol
    r = r_scores(xE, VE)
    check("E r_i = min(x_i, max neighbour x)", np.allclose(r, [0.95, 0.95, 0.85, 0.20]), str(r))
    Y = spillover_curve(r, VE, [0.95, 0.85])
    check("E Y(.95) = 1 - 10/11, Y(.85) = 1 - 11/16", np.allclose(Y, [1 - 10 / 11, 1 - 11 / 16]), str(Y))
    sel = select_threshold(xE, VE)
    check("E theta* = .95 (the only feasible candidate)", sel["theta_star"] == 0.95, str(sel["theta_star"]))
    sel0 = select_threshold(np.array([0.3, 0.4]), np.array([[0, 1.0], [1.0, 0]]))
    check("E empty Theta -> nothing flagged (theta* = inf)", np.isinf(sel0["theta_star"]))

    # ------------------------------------------------------------------ F
    print("\n===== F) toy detection =====")
    rows, wash_leg = [], []
    t = 0
    cycle = [("A", "B"), ("B", "C"), ("C", "A")]
    for rep in range(30):                              # wash triangle, 1x1 clears
        s, b = cycle[rep % 3]
        t += 3
        rows += [(t, s, "SELL", 1), (t, b, "BUY", 1)]
        wash_leg += [True, True]
    # honest traders take one-sided views (D, E only buy; F, G only sell) -> positions never
    # return to zero -> no closures, x0 = 0
    honest = [("D", "F"), ("E", "G"), ("D", "G"), ("E", "F")]
    for rep, (b, s) in enumerate(honest * 3):
        t += 5
        rows += [(t, s, "SELL", 1), (t, b, "BUY", 1)]
        wash_leg += [False, False]
    t += 2
    rows += [(t, "A", "SELL", 1), (t, "D", "BUY", 1)]  # interception: wash leg hit by an honest buy
    wash_leg += [True, False]
    tr = tape(rows)
    det = detect(tr, DetectorConfig())
    labels = dict(trade_legs=pd.DataFrame(dict(leg_id=tr["leg_id"], is_wash_leg=wash_leg)),
                  wallets=pd.DataFrame(dict(wallet=list("ABCDEFG"), is_wash=[True] * 3 + [False] * 4)))
    ev = evaluate(det, labels)
    xw = det.wallets["x"]
    check("F wash wallets score above every honest wallet",
          xw[["A", "B", "C"]].min() > xw[["D", "E", "F", "G"]].max(),
          f"wash min {xw[['A', 'B', 'C']].min():.3f} vs honest max {xw[['D', 'E', 'F', 'G']].max():.3f}")
    check("F Algorithm 2 finds a threshold", np.isfinite(det.theta_star), f"theta*={det.theta_star:.4f}")
    check("F at theta*: trade precision = recall = 1",
          ev["trade_star_precision"] == 1.0 and ev["trade_star_recall"] == 1.0,
          f"P={ev['trade_star_precision']}, R={ev['trade_star_recall']}")
    check("F the interception is not flagged", not det.pairs.loc[
        (det.pairs.buyer == "D") & (det.pairs.seller == "A"), "flag_star"].any())

    # ------------------------------------------------------------------ G
    print("\n===== G) real v3 runs =====")
    tmp = tempfile.mkdtemp()
    try:
        man = generate("mini", seeds=2, backgrounds=("thin",),
                       scenarios=("control", "triad", "triad_camo"), out_dir=tmp, verbose=False)
        root = os.path.join(tmp, "mini")
        rows = []
        for r_ in man.itertuples():
            rdir = os.path.join(root, r_.path)
            # the detector must work with labels/ physically absent
            shutil.move(os.path.join(rdir, "labels"), os.path.join(rdir, "_labels_hidden"))
            det = detect_run(rdir)
            shutil.move(os.path.join(rdir, "_labels_hidden"), os.path.join(rdir, "labels"))
            ev = evaluate_run(rdir, det)
            v = det.V.sum(axis=1)
            cons = det.history @ v
            rows.append(dict(run=r_.run_id, wallets=det.stats["n_wallets"], iters=det.n_iter,
                             amb=det.stats["ambiguous_volume_share"], theta_star=det.theta_star,
                             P_fixed=ev["trade_fixed_precision"], R_fixed=ev["trade_fixed_recall"],
                             FPR_fixed=ev["trade_fixed_fpr"], P_star=ev["trade_star_precision"],
                             R_star=ev["trade_star_recall"],
                             cons=float(np.max(np.abs(cons - cons[0])) / max(cons[0], 1e-12))))
        res = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(res.round(3).to_string(index=False))
        check("G detector runs on every generated run with labels/ hidden", len(res) == len(man))
        check("G Proposition 2 holds on the real graphs", bool((res["cons"] < 1e-9).all()))
        check("G pro-rata pairing is (nearly) exact on this platform",
              bool((res["amb"].fillna(0) < 0.05).all()), f"max ambiguous volume share {res['amb'].max():.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
