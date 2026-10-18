r"""run_smoke.py -- v7 wash-trading smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root:
  python "capstone codes/washtrade/v7/run_smoke.py"

  (A) v6 equivalence     defaults reproduce washtrade.v6 bit for bit (tape, cash, events, LOB) for
                         every v6 scenario on thin and active, incl. microprice and camouflage.
  (B) rotation           rounds draw k distinct pool wallets; each round's events walk its cycle;
                         rotation leaves the event timing unchanged; "lru" spreads use evenly;
                         a bigger pool means more pairs and fewer events per pair.
  (C) size               "units": every wash leg has the base quantity and an edge of size Q is Q
                         events; "block": one event of quantity Q; positions reconstruct.
  (D) residual           planned positions at every round end stay in [-d, d] (all 0 without
                         residuals); in failure-free runs realized positions equal planned ones.
  (E) determinism        same seed -> same run for the full evasive adversary.
  (F) design             every sweep enumerates; sizes match the docstring; shards partition.
  (G) generate + export  sharded runs, resume, merge; tape / slim modes; wash_rounds.csv and
                         rounds_used labels; no leak columns in observable/; schema version.
"""
import os
import sys
import tempfile
from collections import Counter

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

from washtrade.v6 import scenarios as S6
from washtrade.v6.run_smoke import same_run, tape
from washtrade.v7 import analysis as A
from washtrade.v7 import design as D
from washtrade.v7 import scenarios as S
from washtrade.v7.export import SCHEMA_VERSION, export_run, load_labels, load_meta, load_observable
from washtrade.v7.generate import merge, run_sweep


def triad(**kw):
    kw.setdefault("lam_wash", 0.05)
    return [dict(n_wallets=kw.pop("n_wallets", 3), **kw)]


def clean(sim):
    """True if every wash event of the run self-traded (no interception, no failure)."""
    return all(e["self_trade"] for c in sim.controllers for e in c.event_log)


def planned_positions(c):
    """Wallet positions if every logged event had filled as planned."""
    pos = {w: 0 for w in c.wallet_ids}
    for e in c.event_log:
        pos[e["seller"]] -= e["q"]
        pos[e["buyer"]] += e["q"]
    return pos


def boundary_positions(c):
    """Planned positions after the last event of every completed round."""
    pos = {w: 0 for w in c.wallet_ids}
    out = []
    for e in c.event_log:
        pos[e["seller"]] -= e["q"]
        pos[e["buyer"]] += e["q"]
        if e["unit_idx"] + 1 == c.round_log[e["round_idx"]]["n_events"]:
            out.append(dict(pos))
    return out


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) v6 equivalence =====")
    for bg in ("thin", "active"):
        for sc in S6.SCENARIOS:
            for seed in (0, 1):
                a, b = S6.run(bg, sc, seed), S.run(bg, sc, seed)
                check(f"A v7 == v6  {bg:6s} {sc:14s} seed={seed}", same_run(a, b),
                      f"{len(tape(b))} legs")

    # ------------------------------------------------------------------ B
    print("\n===== B) rotation =====")
    for seed in range(3):
        sim = S.run("thin", triad(n_wallets=12, cycle_len=3, rotation="random"), seed)
        c = sim.controllers[0]
        good = True
        for r in c.round_log:
            m = r["members"]
            evs = [e for e in c.event_log if e["round_idx"] == r["round_idx"]]
            walk = [(e["seller"], e["buyer"]) for e in evs]
            want = [(m[i], m[(i + 1) % 3]) for i in range(3)][:len(walk)]
            good &= (len(set(m)) == 3 and set(m) <= set(c.wallet_ids) and walk == want)
        check(f"B seed={seed}: rounds = cycles over 3 distinct pool wallets", good,
              f"{len(c.round_log)} rounds, {sum(v > 0 for v in c.rounds_used.values())}/12 used")
        fixed = S.run("thin", triad(), seed)
        check(f"B seed={seed}: rotation leaves event times unchanged",
              [e["open_time"] for e in c.event_log]
              == [e["open_time"] for e in fixed.controllers[0].event_log])
    lru = S.run("thin", triad(n_wallets=12, cycle_len=3, rotation="lru"), 0).controllers[0]
    uses = list(lru.rounds_used.values())
    check("B lru: rounds per wallet differ by at most 1", max(uses) - min(uses) <= 1, str(uses))
    pairs = {}
    for pool in (3, 12, 96):
        kw = {} if pool == 3 else dict(cycle_len=3, rotation="random")
        g = pd.concat([A.group_summary(S.run("thick", triad(n_wallets=pool, **kw), s))
                       for s in range(3)])
        pairs[pool] = (g["n_pairs"].mean(), g["events_per_pair_mean"].mean())
    check("B bigger pool -> more pairs, fewer events per pair",
          pairs[3][0] < pairs[12][0] < pairs[96][0] and pairs[3][1] > pairs[12][1] > pairs[96][1],
          "  ".join(f"pool {p}: {n:.0f} pairs, {e:.1f} ev/pair" for p, (n, e) in pairs.items()))

    # ------------------------------------------------------------------ C
    print("\n===== C) size =====")
    for seed in range(2):
        sim = S.run("active", triad(q_range=(1, 4)), seed)
        c = sim.controllers[0]
        legs = [mo.order.quantity for mo in sim.markets[0].matched_orders
                if mo.order.order_id >= 1_000_000_000]
        per = Counter((e["round_idx"], e["edge_idx"]) for e in c.event_log)
        full = [r for r in c.round_log if "close_time" in r][:-1]
        check(f"C units seed={seed}: every wash leg has quantity 1", set(legs) == {1})
        check(f"C units seed={seed}: an edge of size Q is Q events",
              all(per[(r["round_idx"], e)] == r["size"] for r in full for e in range(3)),
              f"sizes {Counter(r['size'] for r in c.round_log)}")
        sim = S.run("active", triad(q_range=(1, 4), size_exec="block"), seed)
        c = sim.controllers[0]
        check(f"C block seed={seed}: one event per edge with quantity Q",
              all(e["q"] == c.round_log[e["round_idx"]]["size"] for e in c.event_log))
        recon = A.reconstruct_positions(sim)
        check(f"C block seed={seed}: positions reconstruct from the tape",
              all(recon.get(i, 0) == sim.agents[i].position for i in sim.agents))

    # ------------------------------------------------------------------ D
    print("\n===== D) residual =====")
    n_clean = match = 0
    for d in (1, 2):
        bound = n_res = n_rounds = n_bnd = 0
        for seed in range(12):
            sim = S.run("thin", triad(n_wallets=6, cycle_len=3, rotation="random",
                                      q_range=(1, 3), residual_prob=1.0, residual_max=d), seed)
            c = sim.controllers[0]
            bnd = boundary_positions(c)
            n_bnd += len(bnd)
            bound += all(abs(v) <= d for b in bnd for v in b.values())
            n_res += sum(r["residual"] != 0 for r in c.round_log)
            n_rounds += len(c.round_log)
            if clean(sim):
                n_clean += 1
                match += all(sim.agents[w].position == v for w, v in planned_positions(c).items())
        check(f"D d={d}: planned positions within [-d, d] at every round end", bound == 12,
              f"{n_bnd} round ends, {n_res}/{n_rounds} rounds with a residual")
    check("D failure-free runs: realized position = planned position", n_clean >= 1
          and match == n_clean, f"{match}/{n_clean} failure-free runs")
    shares = [A.group_summary(S.run("thin", triad(residual_prob=p), 0))["residual_round_share"].iloc[0]
              for p in (0.0, 0.5, 1.0)]
    check("D residual share rises with residual_prob (0 at p = 0)",
          shares[0] == 0 and shares[0] < shares[1] < shares[2], str(np.round(shares, 2).tolist()))
    zero = n_bnd = 0
    for seed in range(6):
        sim = S.run("thin", triad(n_wallets=12, cycle_len=3, rotation="random", q_range=(1, 4)), seed)
        bnd = boundary_positions(sim.controllers[0])
        n_bnd += len(bnd)
        zero += all(v == 0 for b in bnd for v in b.values())
    check("D no residual -> every planned position is 0 at every round end", zero == 6,
          f"{n_bnd} round ends")
    nz = {}
    for p in (0.0, 1.0):
        vals = []
        for seed in range(6):
            sim = S.run("thin", triad(residual_prob=p), seed)
            if clean(sim):
                vals.append(A.group_summary(sim)["round_end_nonzero_share"].iloc[0])
        nz[p] = vals
    check("D realized round-end positions: all 0 without residuals, not all 0 with p = 1",
          nz[0.0] and nz[1.0] and max(nz[0.0]) == 0 and min(nz[1.0]) > 0,
          f"p=0: {np.round(nz[0.0], 2).tolist()}  p=1: {np.round(nz[1.0], 2).tolist()}")

    # ------------------------------------------------------------------ E
    print("\n===== E) determinism =====")
    for sc in ("triad_evasive", "evasive_camo"):
        a, b = S.run("active", sc, 3), S.run("active", sc, 3)
        check(f"E {sc}: same seed -> identical run", same_run(a, b)
              and a.controllers[0].round_log == b.controllers[0].round_log)

    # ------------------------------------------------------------------ F
    print("\n===== F) design =====")
    for sw in D.SWEEPS:
        runs = D.enumerate_runs(sw, seeds=3)
        ids = [r.run_id for r in runs]
        parts = [D.shard(runs, i, 7) for i in range(7)]
        check(f"F {sw}: {len(runs)} runs, unique ids, 7 shards partition them",
              len(set(ids)) == len(ids)
              and sorted(r.run_id for p in parts for r in p) == sorted(ids))
        check(f"F {sw}: {D.SWEEP_CELLS[sw]} cells, as in the docstring",
              len(D.enumerate_runs(sw, seeds=1)) == D.SWEEP_CELLS[sw])
    for spec in D.enumerate_runs("evasive", seeds=1)[:40]:
        S.resolve_groups(spec.groups)             # every group dict is a valid v7 WashGroup
    check("F evasive group specs build", True)

    # ------------------------------------------------------------------ G
    print("\n===== G) generate + export =====")
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(seeds=1, out_dir=tmp, name="mini", limit=3, verbose=False)
        d0 = run_sweep("rotation", shard=0, n_shards=2, **kw)
        d1 = run_sweep("rotation", shard=1, n_shards=2, **kw)
        again = run_sweep("rotation", shard=0, n_shards=2, **kw)
        check("G two shards run disjoint runs", d0[0] == 3 and d1[0] == 3 and d0[2] == d1[2] == 0,
              f"{d0} {d1}")
        check("G a re-run skips finished runs", again == (0, 3, 0), str(again))
        man = merge("rotation", out_dir=tmp, name="mini", verbose=False)
        check("G merge collects every finished run", len(man) == 6
              and {"f_pool", "f_cycle_len", "n_pairs", "honest_ever_zero_share"} <= set(man.columns),
              str(len(man)))

        forbidden = {"order_id", "agent_id", "is_wash", "is_wash_leg", "is_wash_wallet",
                     "group_id", "event_idx", "round_idx", "fundamental", "wash_legs",
                     "wash_posts", "wash_phase", "camouflage", "rounds_used"}
        for mode in ("slim", "tape"):
            sim = S.run("active", "evasive_camo", 1, record_lob=(mode != "tape"))
            rdir = os.path.join(tmp, mode)
            export_run(sim, rdir, run_id=f"x_{mode}", seed=1, scenario="evasive_camo",
                       background="active", mode=mode)
            obs, lab, meta = load_observable(rdir), load_labels(rdir), load_meta(rdir)
            leaks = {n: sorted(forbidden & set(df.columns)) for n, df in obs.items()}
            check(f"G {mode}: observable carries no label columns",
                  all(not v for v in leaks.values()), str(leaks))
            w, rounds, ev = lab["wallets"], lab["wash_rounds"], lab["wash_events"]
            members = {m for s in rounds["members"] for m in s.split(";")}
            wash_codes = set(w.loc[w["is_wash"], "wallet"])
            check(f"G {mode}: wash_rounds members are wash wallets of the group",
                  members <= wash_codes and len(rounds) == meta["summary"]["n_rounds"])
            check(f"G {mode}: rounds_used on wash wallets only, sums to k x rounds",
                  bool(w.loc[~w["is_wash"], "rounds_used"].isna().all())
                  and int(w["rounds_used"].sum()) == 3 * len(rounds))
            check(f"G {mode}: wash_events carry round_idx / unit_idx",
                  {"round_idx", "unit_idx", "piece"} <= set(ev.columns))
            check(f"G {mode}: schema {SCHEMA_VERSION}, v7 metrics in meta, files listed",
                  meta["schema_version"] == SCHEMA_VERSION and "n_pairs" in meta["groups"][0]
                  and "labels/wash_rounds.csv" in meta["files"]
                  and ("observable/lob.csv" in meta["files"]) == (mode != "tape"))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
