r"""run_smoke.py -- v8 wash-trading smoke test (headless). Ends with "RESULT: ALL PASS".

Run from the repo root:
  python "capstone codes/washtrade/v8/run_smoke.py"

  (A) v7 equivalence     defaults reproduce washtrade.v7 bit for bit (tape, cash, events, LOB,
                         rounds) for every v7 scenario on thin and active.
  (B) multi-unit honest  m = 1 reproduces v7; a k-unit order is priced at the mean of its k
                         marginal values (k = 1: ZI's price); with m = 4 every drawn size is in
                         1..4, honest legs are <= 4, positions reconstruct from the tape, camouflage
                         wallets trade multi-unit too, and the wash event times do not change.
  (C) edge order         "interleaved" walks a round's edges round robin; it changes neither the
                         rounds (members, size, residual) nor the event times.
  (D) pipeline           D rounds open at once, taken in turn; a wallet never has legs in two open
                         rounds; event times unchanged; holding time grows with D; bad settings raise.
  (E) determinism        same seed -> same run for the strongest adversary on a multi-unit market.
  (F) design             every sweep enumerates; sizes match the docstring; shards partition; every
                         group spec builds.
  (G) generate + export  sharded runs, resume, merge; schema version and v8 metrics; no label
                         columns in observable/; detection v4 reads a multi-unit run.
"""
import os
import sys
import tempfile
from collections import Counter, defaultdict

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

from marketsim.fourheap.constants import BUY, SELL
from washtrade.v6.run_smoke import same_run
from washtrade.v7 import scenarios as S7
from washtrade.v8 import analysis as A
from washtrade.v8 import design as D
from washtrade.v8 import scenarios as S
from washtrade.v8.agents import MultiUnitZI
from washtrade.v8.export import SCHEMA_VERSION, export_run, load_labels, load_meta, load_observable
from washtrade.v8.generate import merge, run_sweep

WASH_OID = 1_000_000_000


def group(**kw):
    kw.setdefault("lam_wash", 0.05)
    return [dict(n_wallets=kw.pop("n_wallets", 3), **kw)]


def evasive(**kw):
    return group(n_wallets=kw.pop("n_wallets", 24), cycle_len=3, rotation="random",
                 q_range=(1, 4), **kw)


def times(sim):
    return [e["open_time"] for e in sim.controllers[0].event_log]


def rounds(sim):
    return [(tuple(r["members"]), r["size"], r["residual"]) for r in sim.controllers[0].round_log]


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) v7 equivalence =====")
    for bg in ("thin", "active"):
        bad = []
        for sc in S7.SCENARIOS:
            a, b = S7.run(bg, sc, 0), S.run(bg, sc, 0)
            same = same_run(a, b) and all(ca.round_log == cb.round_log
                                          for ca, cb in zip(a.controllers, b.controllers))
            if not same:
                bad.append(sc)
        check(f"A v8 == v7 on {bg}: all {len(S7.SCENARIOS)} v7 scenarios", not bad, str(bad))

    # ------------------------------------------------------------------ B
    print("\n===== B) multi-unit honest orders =====")
    for seed in (0, 1):
        a = S7.run("active", "evasive_camo", seed)
        b = S.run(dict(S.BACKGROUNDS["active"], order_size_max=1, name="m1"), "evasive_camo", seed)
        check(f"B m = 1 reproduces v7 (active, evasive_camo, seed {seed})", same_run(a, b))
    sim = S.run("active_mu4", "evasive_camo", 0)
    mz = [getattr(ag, "_agent", ag) for ag in sim.agents.values()]
    mz = [z for z in mz if isinstance(z, MultiUnitZI)]
    z = mz[0]
    saved, z.position = z.position, 2
    v3 = np.mean([z.pv.value_for_exchange(2 + i, BUY) for i in range(3)])
    check("B k-unit price = mean of k marginal values; k = 1 is ZI's",
          np.isclose(z._unit_value(BUY, 3), v3)
          and z._unit_value(SELL, 1) == z.pv.value_for_exchange(2, SELL)
          and np.isclose(z._unit_value(SELL, 2),
                         np.mean([z.pv.value_for_exchange(2 - i, SELL) for i in range(2)])))
    z.position = saved
    sizes = sim.honest_order_sizes()
    cnt = Counter(sizes)
    check("B m = 4: every drawn size in 1..4, all four used", set(cnt) == {1, 2, 3, 4},
          str(dict(sorted(cnt.items()))))
    honest_q = [mo.order.quantity for mo in sim.markets[0].matched_orders
                if mo.order.order_id < WASH_OID]
    check("B honest legs have quantity <= 4, some above 1",
          max(honest_q) <= 4 and max(honest_q) > 1, f"max {max(honest_q)}")
    good = True
    for s in range(3):
        sm = S.run("active_mu4", "evasive_camo", s)
        rec = A.reconstruct_positions(sm)
        good &= all(rec.get(i, 0) == sm.agents[i].position for i in sm.agents)
    check("B positions reconstruct from the tape (3 seeds)", good)
    camo = [getattr(sim.agents[a], "_agent", None) for a in sim.camouflage_ids]
    check("B camouflage wallets are multi-unit traders too",
          camo and all(isinstance(c, MultiUnitZI) and c.order_size_max == 4 for c in camo),
          f"{len(camo)} camouflage wallets")
    plain = S.run("active", "evasive_camo", 0)
    check("B multi-unit background leaves the wash event times unchanged",
          times(plain) == times(sim), f"{len(times(sim))} events")

    # ------------------------------------------------------------------ C
    print("\n===== C) edge order =====")
    for seed in (0, 1):
        seq = S.run("thick", evasive(residual_prob=0.5), seed)
        itl = S.run("thick", evasive(residual_prob=0.5, edge_order="interleaved"), seed)
        c = itl.controllers[0]
        walk = True
        for r in c.round_log:
            evs = sorted((e for e in c.event_log if e["round_idx"] == r["round_idx"]),
                         key=lambda e: e["unit_idx"])
            edges = [e["edge_idx"] for e in evs]
            k = r["cycle_len"]
            full = min(r["size"], r["size"] + r["residual"])     # units every edge has
            want = [u % k for u in range(full * k)]
            walk &= edges[:len(want)] == want[:len(edges)]
            walk &= [e["unit_idx"] for e in evs] == list(range(len(evs)))
        check(f"C seed {seed}: interleaved rounds walk their edges round robin", walk,
              f"{len(c.round_log)} rounds")
        check(f"C seed {seed}: same rounds (members, size, residual) and event times",
              rounds(seq) == rounds(itl) and times(seq) == times(itl))
    rs = [A.repeat_share(S.run("thick", evasive(edge_order=o), s).controllers[0].event_log)
          for o in ("sequential", "interleaved") for s in range(3)]
    check("C interleaving lowers the share of same-pair repeats within 50 steps",
          np.mean(rs[3:]) < np.mean(rs[:3]), f"{np.mean(rs[:3]):.2f} -> {np.mean(rs[3:]):.2f}")

    # ------------------------------------------------------------------ D
    print("\n===== D) pipeline =====")
    holds = {}
    for d in (1, 4, 8):
        h = []
        for seed in range(3):
            sim = S.run("thick", evasive(pipeline=d, n_wallets=96), seed)
            c = sim.controllers[0]
            if d > 1:
                ev = c.event_log
                spans = defaultdict(list)                  # wallet -> [(first, last) per round]
                idx = defaultdict(list)
                for i, e in enumerate(ev):
                    for w in (e["seller"], e["buyer"]):
                        idx[(w, e["round_idx"])].append(i)
                for (w, r), ii in idx.items():
                    spans[w].append((min(ii), max(ii)))
                disjoint = all(b[0] > a[1] for s in spans.values()
                               for a, b in zip(sorted(s), sorted(s)[1:]))
                rr = [e["round_idx"] for e in ev[:d]]
                check(f"D D={d} seed {seed}: {d} rounds open, first {d} events from {d} rounds, "
                      "no wallet in two open rounds",
                      c.max_open_rounds == d and len(set(rr)) == d and disjoint)
                base = S.run("thick", evasive(n_wallets=96), seed)
                check(f"D D={d} seed {seed}: event times unchanged", times(base) == times(sim))
            h.append(A.group_summary(sim)["hold_mean"].iloc[0])
        holds[d] = float(np.nanmean(h))
    check("D mean holding time grows with the pipeline depth",
          holds[1] < holds[4] < holds[8], "  ".join(f"D={d}: {v:.0f}" for d, v in holds.items()))
    for bad, why in ((dict(pipeline=2), "fixed pool"), (dict(pipeline=40), "pool too small"),
                     (dict(pipeline=2, unit_lam=0.5), "unit_lam")):
        try:
            g = group(**bad) if why == "fixed pool" else evasive(**bad)
            S.run("thin", g, 0)
            raised = False
        except ValueError:
            raised = True
        check(f"D invalid setting raises ({why})", raised)

    # ------------------------------------------------------------------ E
    print("\n===== E) determinism =====")
    a, b = S.run("active_mu4", "evasive96_ip8", 3), S.run("active_mu4", "evasive96_ip8", 3)
    check("E evasive96_ip8 on active_mu4: same seed -> identical run", same_run(a, b)
          and rounds(a) == rounds(b) and a.honest_order_sizes() == b.honest_order_sizes())

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
        for spec in D.enumerate_runs(sw, seeds=1):
            S.resolve_groups(spec.groups)
            S.resolve_background(dict(spec.background_spec, name=spec.background))
        check(f"F {sw}: every group and background spec builds", True)

    # ------------------------------------------------------------------ G
    print("\n===== G) generate + export =====")
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(seeds=1, out_dir=tmp, name="mini", limit=3, verbose=False)
        d0 = run_sweep("honest", shard=0, n_shards=2, **kw)
        d1 = run_sweep("honest", shard=1, n_shards=2, **kw)
        again = run_sweep("honest", shard=0, n_shards=2, **kw)
        check("G two shards run disjoint runs", d0[0] == 3 and d1[0] == 3 and d0[2] == d1[2] == 0,
              f"{d0} {d1}")
        check("G a re-run skips finished runs", again == (0, 3, 0), str(again))
        man = merge("honest", out_dir=tmp, name="mini", verbose=False)
        check("G merge collects every finished run with v8 fields", len(man) == 6
              and {"f_honest_size", "f_scenario", "honest_mean_order_size"} <= set(man.columns),
              str(len(man)))

        forbidden = {"order_id", "agent_id", "is_wash", "is_wash_leg", "is_wash_wallet",
                     "group_id", "event_idx", "round_idx", "fundamental", "camouflage",
                     "rounds_used"}
        sim = S.run("active_mu4", [dict(D._evasive(edge_order="interleaved", pipeline=8))], 1,
                    record_lob=False)
        rdir = os.path.join(tmp, "one")
        export_run(sim, rdir, run_id="x", seed=1, scenario="ip8", background="active_mu4",
                   mode="tape")
        obs, lab, meta = load_observable(rdir), load_labels(rdir), load_meta(rdir)
        leaks = {n: sorted(forbidden & set(df.columns)) for n, df in obs.items()}
        check("G observable carries no label columns", all(not v for v in leaks.values()),
              str(leaks))
        g = meta["groups"][0]
        check(f"G schema {SCHEMA_VERSION}, v8 metrics in meta",
              meta["schema_version"] == SCHEMA_VERSION
              and {"hold_mean", "repeat_share_50", "pipeline", "edge_order"} <= set(g)
              and meta["summary"]["honest_mean_order_size"] > 1 and g["pipeline"] == 8)
        check("G tape has multi-unit honest legs", obs["trades"]["qty"].max() > 1)
        from detection.v4.detector import detect_v4
        det = detect_v4(obs["trades"])
        check("G detection v4 runs on a multi-unit run",
              len(det.pairs) > 0 and np.isfinite(det.pairs["closure"].dropna()).all())

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
