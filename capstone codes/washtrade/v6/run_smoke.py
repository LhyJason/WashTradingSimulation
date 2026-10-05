r"""run_smoke.py -- v6 wash-trading smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root:
  python "capstone codes/washtrade/v6/run_smoke.py"

  (A) v3 equivalence        defaults reproduce washtrade.v3 bit for bit (tape, cash, events, LOB)
                            for every v3 scenario on thin and thick.
  (B) v5 equivalence        price_rule="microprice" reproduces washtrade.v5 bit for bit.
  (C) background knob r     r reaches the fundamental; with r = 1e-3 honest trading covers the
                            whole run, with r = 0.01 the middle of the run is (almost) silent.
  (D) camouflage intensity  own-rate wallets do not touch the shared arrival stream (background
                            arrival schedule identical across rates and to the no-camouflage
                            run); ordinary trading grows with the rate; no live wash leg is
                            cancelled while its event is open; positions reconstruct.
  (E) partial camouflage    exactly n_camouflage wallets camouflage; n_camouflage = n_wallets
                            reproduces v3's camouflage=True.
  (F) leakage metric        in [0, 1]; <= 0.05 for a gap-0 triad on thin; grows with camouflage.
  (G) design                every sweep enumerates; run ids unique; shards partition the runs.
  (H) generate + export     sharded runs, resume skips finished runs, merge, export modes,
                            no leak columns in observable/, schema version.
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

from washtrade.v3 import scenarios as S3
from washtrade.v5 import scenarios as S5
from washtrade.v6 import analysis as A
from washtrade.v6 import design as D
from washtrade.v6 import scenarios as S
from washtrade.v6.export import SCHEMA_VERSION, export_run, load_labels, load_meta, load_observable
from washtrade.v6.generate import merge, run_sweep


def tape(sim):
    return [(m.time, m.order.agent_id, m.order.order_id, m.order.order_type,
             m.order.quantity, m.price) for m in sim.markets[0].matched_orders]


def same_run(a, b):
    """Tape, cash, positions, every controller's event log and the LOB snapshots."""
    if tape(a) != tape(b):
        return False
    if any(a.agents[i].position != b.agents[i].position or a.agents[i].cash != b.agents[i].cash
           for i in a.agents) or set(a.agents) != set(b.agents):
        return False
    if len(a.controllers) != len(b.controllers):
        return False
    for ca, cb in zip(a.controllers, b.controllers):
        ea, eb = pd.DataFrame(ca.event_log), pd.DataFrame(cb.event_log)
        if len(ea) != len(eb) or (len(ea) and not ea.equals(eb[ea.columns])):
            return False
    return a.lob.snapshots_df().equals(b.lob.snapshots_df())


def live_wash_cancels(sim):
    """Run ``sim`` and count cancellations of a still-resting wash leg whose event was open."""
    ob = sim.markets[0].order_book
    removed = []
    orig = ob.remove

    def logged(oid):
        live = ob.buy_unmatched.contains(oid) or ob.sell_unmatched.contains(oid)
        removed.append((sim.time, oid, live))
        return orig(oid)
    ob.remove = logged
    sim.run()
    bad = 0
    for t, oid, live in removed:
        m = sim.wash_meta(oid)
        if m is None or not live:
            continue
        if t <= sim.controllers[m["group_id"]].event_log[m["event_idx"]]["close_time"]:
            bad += 1
    return bad


def background_arrivals(sim):
    """Run ``sim`` while logging (time, agent_id) of every background (non-wallet) arrival."""
    log = []
    wallets = set(sim.wash_wallet_ids)
    for aid, agent in sim.agents.items():
        if aid in wallets:
            continue
        orig = agent.take_action

        def logged(_orig=orig, _aid=aid):
            log.append((sim.time, _aid))
            return _orig()
        agent.take_action = logged
    sim.run()
    return log


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) v3 equivalence =====")
    for bg in ("thin", "thick"):
        for sc in S3.SCENARIOS:
            for seed in (0, 1):
                a, b = S3.run(bg, sc, seed), S.run(bg, sc, seed)
                check(f"A v6 == v3  {bg:5s} {sc:13s} seed={seed}", same_run(a, b),
                      f"{len(tape(b))} legs")

    # ------------------------------------------------------------------ B
    print("\n===== B) v5 equivalence (price_rule='microprice') =====")
    for bg in ("thin", "thick"):
        for sc, groups in (("triad", [dict(n_wallets=3, lam_wash=0.05)]),
                           ("triad_gap5", [dict(n_wallets=3, lam_wash=0.05, gap=5)]),
                           ("hard", [dict(n_wallets=3, lam_wash=0.05, gap=5, camouflage=True),
                                     dict(n_wallets=2, lam_wash=0.05, gap=5, camouflage=True)])):
            for seed in (0, 2):
                a = S5.run(bg, sc, seed)
                b = S.run(bg, [dict(g, price_rule="microprice") for g in groups], seed)
                check(f"B v6(microprice) == v5  {bg:5s} {sc:10s} seed={seed}", same_run(a, b),
                      f"{len(tape(b))} legs")

    # ------------------------------------------------------------------ C
    print("\n===== C) background knob r =====")
    for bg, r in (("thick", 0.01), ("active", 1e-3)):
        sim = S.run(bg, "control", 0)
        check(f"C {bg}: fundamental r = {r}",
              abs(sim.markets[0].fundamental.get_r() - r) < 1e-9 and sim.run_config["r"] == r)
    mids_slow, mids_fast, bins_slow = [], [], []
    for seed in range(3):
        fast = A.background_activity(S.run("thick", "control", seed))
        slow = A.background_activity(S.run("active", "control", seed))
        mids_fast.append(fast["background_middle_share"])
        mids_slow.append(slow["background_middle_share"])
        bins_slow.append(slow["background_active_bins"])
    check("C r = 1e-3: honest trades in every tenth of the run", min(bins_slow) == 10, str(bins_slow))
    check("C r = 1e-3: >= 30% of honest legs in the middle half",
          min(mids_slow) >= 0.30, f"{np.round(mids_slow, 2).tolist()}")
    check("C r = 0.01: < 10% of honest legs in the middle half (the silent middle)",
          max(mids_fast) < 0.10, f"{np.round(mids_fast, 2).tolist()}")

    # ------------------------------------------------------------------ D
    print("\n===== D) camouflage intensity =====")
    for bg in ("thin", "active"):
        sched = {}
        for label, groups in (
                ("none", [dict(n_wallets=3, lam_wash=0.05)]),
                ("lam=.005", [dict(n_wallets=3, lam_wash=0.05, camouflage=True, camouflage_lam=5e-3)]),
                ("lam=.05", [dict(n_wallets=3, lam_wash=0.05, camouflage=True, camouflage_lam=5e-2)])):
            sched[label] = background_arrivals(S.build(bg, groups, 4))
        check(f"D {bg}: background arrival schedule identical across camouflage rates",
              sched["none"] == sched["lam=.005"] == sched["lam=.05"],
              f"{len(sched['none'])} background arrivals")
    legit = {}
    for rate in (5e-3, 5e-2):
        tot = 0
        for seed in range(4):
            sim = S.build("active", [dict(n_wallets=3, lam_wash=0.05, gap=5, camouflage=True,
                                          camouflage_lam=rate)], seed)
            bad = live_wash_cancels(sim)
            s = A.summarize(sim)
            tot += s["camouflage_legit_legs"]
            check(f"D rate={rate} seed={seed}: no live wash leg cancelled while its event is open",
                  bad == 0, f"{s['n_events']} events, {s['camouflage_legit_legs']} ordinary legs")
            recon = A.reconstruct_positions(sim)
            check(f"D rate={rate} seed={seed}: positions reconstruct from the tape",
                  all(recon.get(i, 0) == sim.agents[i].position for i in sim.agents))
            check(f"D rate={rate} seed={seed}: wallets on private arrival streams",
                  set(sim.own_arrivals) == set(sim.camouflage_ids) == set(sim.wash_wallet_ids))
        legit[rate] = tot
    check("D ordinary trading grows with the camouflage rate (pooled over 4 seeds)",
          legit[5e-2] > 2 * legit[5e-3], str(legit))

    # ------------------------------------------------------------------ E
    print("\n===== E) partial camouflage =====")
    sim = S.run("thin", [dict(n_wallets=3, lam_wash=0.05, camouflage=True, n_camouflage=1)], 0)
    check("E n_camouflage=1: exactly one camouflage wallet",
          len(sim.camouflage_ids) == 1 and sim.camouflage_ids[0] == sim.wash_wallet_ids[0])
    for seed in (0, 1):
        a = S3.run("thin", "hard", seed)
        b = S.run("thin", [dict(n_wallets=3, lam_wash=0.05, gap=5, camouflage=True, n_camouflage=3),
                           dict(n_wallets=2, lam_wash=0.05, gap=5, camouflage=True, n_camouflage=2)],
                  seed)
        check(f"E n_camouflage = n_wallets == v3 camouflage=True (hard, seed={seed})", same_run(a, b))

    # ------------------------------------------------------------------ F
    print("\n===== F) leakage metric =====")
    leaks = {}
    for sc in ("triad", "triad_camo_hi", "hard_hi"):
        vals = []
        for seed in range(3):
            g = A.group_summary(S.run("thin", sc, seed))
            vals += g["leak_prorata"].tolist() + g["leak_maxmatch"].tolist()
        leaks[sc] = vals
    check("F leakage in [0, 1]", all(0 <= v <= 1 for v in sum(leaks.values(), [])))
    check("F thin triad gap 0: leakage <= 0.05", max(leaks["triad"]) <= 0.05,
          f"max {max(leaks['triad']):.3f}")
    check("F camouflage raises leakage", np.mean(leaks["triad_camo_hi"]) > np.mean(leaks["triad"]),
          f"triad {np.mean(leaks['triad']):.3f} vs camo_hi {np.mean(leaks['triad_camo_hi']):.3f}")

    # ------------------------------------------------------------------ G
    print("\n===== G) design =====")
    for sw in D.SWEEPS:
        runs = D.enumerate_runs(sw, seeds=3)
        ids = [r.run_id for r in runs]
        parts = [D.shard(runs, i, 7) for i in range(7)]
        check(f"G {sw}: {len(runs)} runs, unique ids, 7 shards partition them",
              len(set(ids)) == len(ids)
              and sorted(r.run_id for p in parts for r in p) == sorted(ids))
    check("G sweep sizes at 20 seeds match the docstring table",
          [len(D.enumerate_runs(sw)) for sw in D.SWEEPS] == [1300, 1440, 1200, 1560, 600, 360])

    # ------------------------------------------------------------------ H
    print("\n===== H) generate + export =====")
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(seeds=1, out_dir=tmp, name="mini", limit=3, verbose=False)
        d0 = run_sweep("price", shard=0, n_shards=2, **kw)
        d1 = run_sweep("price", shard=1, n_shards=2, **kw)
        again = run_sweep("price", shard=0, n_shards=2, **kw)
        check("H two shards run disjoint runs", d0[0] == 3 and d1[0] == 3 and d0[2] == d1[2] == 0,
              f"{d0} {d1}")
        check("H a re-run skips finished runs", again == (0, 3, 0), str(again))
        man = merge("price", out_dir=tmp, name="mini", verbose=False)
        check("H merge collects every finished run", len(man) == 6
              and {"f_gap", "f_price_rule", "leak_prorata", "background_legs_per_agent"}
              <= set(man.columns))
        root = os.path.join(tmp, "mini")
        check("H no leftover .tmp folders",
              not any(f.endswith(".tmp") for f in os.listdir(os.path.join(root, "runs"))))

        forbidden = {"order_id", "agent_id", "is_wash", "is_wash_leg", "is_wash_wallet",
                     "group_id", "event_idx", "fundamental", "fund_stale", "wash_legs",
                     "wash_posts", "wash_phase", "camouflage", "camouflage_lam"}
        for mode in ("full", "slim", "tape"):
            sim = S.run("active", "hard_hi", 1, record_lob=(mode != "tape"))
            rdir = os.path.join(tmp, mode)
            export_run(sim, rdir, run_id=f"x_{mode}", seed=1, scenario="hard_hi",
                       background="active", mode=mode)
            obs, lab, meta = load_observable(rdir), load_labels(rdir), load_meta(rdir)
            files = set(meta["files"])
            expect_lob = mode != "tape"
            check(f"H {mode}: LOB files present iff mode != tape",
                  ("observable/lob.csv" in files) == expect_lob
                  and ("observable/lob_levels.csv" in files) == (mode == "full")
                  and os.path.exists(os.path.join(rdir, "observable", "lob.csv")) == expect_lob)
            leaks_ = {n: sorted(forbidden & set(df.columns)) for n, df in obs.items()
                      if n != "lob" or expect_lob}
            check(f"H {mode}: observable carries no label / leak columns",
                  all(not v for v in leaks_.values()), str(leaks_))
            w = lab["wallets"]
            check(f"H {mode}: wallets.csv has camouflage_lam on exactly the camouflage wallets",
                  bool((w["camouflage_lam"].notna() == w["camouflage"]).all()))
            check(f"H {mode}: schema {SCHEMA_VERSION} + v6 summary in meta",
                  meta["schema_version"] == SCHEMA_VERSION and "leak_prorata" in meta["summary"]
                  and meta["export_mode"] == mode)

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
