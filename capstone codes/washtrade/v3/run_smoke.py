r"""run_smoke.py -- v3 wash-trading smoke test (headless).

Run (from the repo root):
  python "capstone codes/washtrade/v3/run_smoke.py"

  (A) all-ZI control                 -> no groups, market still trades, LOB recorded.
  (B) v2 equivalence                 -> single group, no camouflage: identical tape, cash,
                                        event log and LOB to washtrade.v2 (gap 0 / 5 / 20).
  (C) camouflage                     -> wallets make ordinary ZI trades AND wash trades; an
                                        exposed wash leg is never cancelled while its event is
                                        open (checked on every order_book.remove call).
  (D) multiple groups                -> disjoint wallets and order_id ranges, no two groups
                                        open in the same timestep, every group trades.
  (E) export schema                  -> observable files carry no label/leak columns, wallets are
                                        anonymized, labels join back to the exact positions.
  (F) dataset generation             -> manifest + one run folder per (background, scenario, seed).
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

from washtrade.v3 import WashGroup, WashTradingSimulator
from washtrade.v3 import analysis as A
from washtrade.v3 import scenarios as S
from washtrade.v3.controller import WASH_OID_FLOOR, WashController, is_wash_oid
from washtrade.v3.export import export_run, load_labels, load_observable, load_meta
from washtrade.v3.generate import generate

LAM_WASH = 0.05


def run_v(cls, comp, seed, **kw):
    S.seed_all(seed)
    sim = cls(comp, sim_time=2000, lam=5e-3, wash_seed=seed, **kw)
    sim.run()
    return sim


def tape(sim):
    return [(m.time, m.order.agent_id, m.order.order_id, m.order.order_type,
             m.order.quantity, m.price) for m in sim.markets[0].matched_orders]


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ------------------------------------------------------------------ A
    print("\n===== A) control =====")
    ctrl = S.run("thin", "control", 0)
    check("A control has no wash groups", len(ctrl.controllers) == 0)
    check("A control still trades", len(ctrl.markets[0].matched_orders) > 0,
          f"{len(ctrl.markets[0].matched_orders)} legs")
    check("A control records a LOB", len(A.lob_df(ctrl)) > 0)

    # ------------------------------------------------------------------ B
    print("\n===== B) v2 equivalence =====")
    from washtrade.v2 import WashTradingSimulator as V2
    cases = [({"ZI": 14}, {}),
             ({"ZI": 13, "WashTrader": 3}, dict(lam_wash=LAM_WASH, wash_gap=0)),
             ({"ZI": 13, "WashTrader": 3}, dict(lam_wash=LAM_WASH, wash_gap=5)),
             ({"ZI": 13, "WashTrader": 2}, dict(lam_wash=LAM_WASH, wash_gap=20))]
    for comp, kw in cases:
        for seed in (0, 3):
            a, b = run_v(V2, comp, seed, **kw), run_v(WashTradingSimulator, comp, seed, **kw)
            ev_a = pd.DataFrame(a.controller.event_log)
            ev_b = pd.DataFrame(b.controllers[0].event_log) if b.controllers else pd.DataFrame()
            la, lb = a.lob.snapshots_df(), b.lob.snapshots_df()
            cols = [c for c in la.columns if c in lb.columns and c not in ("wash_legs", "wash_phase")]
            same = (tape(a) == tape(b)
                    and all(a.agents[i].position == b.agents[i].position
                            and a.agents[i].cash == b.agents[i].cash for i in a.agents)
                    and len(ev_a) == len(ev_b)
                    and (len(ev_a) == 0 or ev_a.equals(ev_b[ev_a.columns]))
                    and la[cols].equals(lb[cols]))
            check(f"B v3 == v2  {comp} {kw} seed={seed}", same,
                  f"{len(tape(b))} legs, {len(ev_b)} events")

    # ------------------------------------------------------------------ C
    print("\n===== C) camouflage =====")
    camo_legit_total = 0
    for seed in range(4):
        S.seed_all(seed)
        sim = WashTradingSimulator({"ZI": 13}, wash_groups=[
            WashGroup(3, gap=5, lam_wash=LAM_WASH, camouflage=True)],
            wash_seed=seed, sim_time=2000, lam=5e-3)
        ob = sim.markets[0].order_book
        removed = []
        _orig = ob.remove

        def _logged(oid, _o=_orig, _ob=ob, _sim=sim, _log=removed):
            live = _ob.buy_unmatched.contains(oid) or _ob.sell_unmatched.contains(oid)
            _log.append((_sim.time, oid, live))
            return _o(oid)

        ob.remove = _logged
        sim.run()
        bad = 0
        for t, oid, live in removed:
            m = sim.wash_meta(oid)
            if m is None or not live:
                continue
            if t <= sim.controllers[0].event_log[m["event_idx"]]["close_time"]:
                bad += 1
        s = A.summarize(sim)
        camo_legit_total += s["camouflage_legit_legs"]
        check(f"C seed={seed}: no live wash leg cancelled while its event is open", bad == 0,
              f"{s['n_events']} events, {s['camouflage_legit_legs']} ordinary legs by wallets")
        recon = A.reconstruct_positions(sim)
        check(f"C seed={seed}: positions reconstruct from the tape",
              all(recon.get(i, 0) == sim.agents[i].position for i in sim.agents))
        check(f"C seed={seed}: camouflage wallets are in the arrival stream",
              set(sim.camouflage_ids) == set(sim.wash_wallet_ids))
    check("C camouflage wallets do trade ordinarily (pooled over seeds)", camo_legit_total > 0,
          f"{camo_legit_total} ordinary legs")

    # ------------------------------------------------------------------ D
    print("\n===== D) multiple groups =====")
    for scen in ("two_groups", "hard"):
        sim = S.run("thin", scen, 0)
        ids = [set(c.wallet_ids) for c in sim.controllers]
        check(f"D {scen}: wallet sets disjoint", len(set().union(*ids)) == sum(len(x) for x in ids))
        ranges_ok = all(
            (oid - WASH_OID_FLOOR) // WashController._OID_GROUP_STRIDE == c.group_id
            for c in sim.controllers for oid in c.oid_meta)
        check(f"D {scen}: order_id ranges attribute every leg to its group", ranges_ok)
        ev = A.event_log_df(sim)
        dup_open = int((ev.groupby("open_time")["group_id"].nunique() > 1).sum())
        check(f"D {scen}: no two groups open in the same timestep", dup_open == 0,
              f"deferred opens: {[c.n_deferred for c in sim.controllers]}")
        gs = A.group_summary(sim)
        check(f"D {scen}: every group fires events", bool((gs["n_events"] > 0).all()),
              ", ".join(f"g{r.group_id}: {r.n_events} ev / self {r.self_trade_rate:.2f}"
                        for r in gs.itertuples()))
    tg = S.run("thin", "two_groups", 0)
    check("D two_groups (gap 0): both groups self-trade ~always",
          bool((A.group_summary(tg)["self_trade_rate"] > 0.9).all()))

    # ------------------------------------------------------------------ E
    print("\n===== E) export schema =====")
    forbidden = {"order_id", "agent_id", "is_wash", "is_wash_leg", "is_wash_wallet", "group_id",
                 "event_idx", "fundamental", "fund_stale", "wash_legs", "wash_posts", "wash_phase"}
    with tempfile.TemporaryDirectory() as tmp:
        sim = S.run("thin", "hard", 1)
        rdir = os.path.join(tmp, "run")
        export_run(sim, rdir, run_id="thin__hard__s001", seed=1, scenario="hard", background="thin")
        obs, lab, meta = load_observable(rdir), load_labels(rdir), load_meta(rdir)
        leaks = {name: sorted(forbidden & set(df.columns)) for name, df in obs.items()}
        check("E observable files carry no label / leak columns",
              all(not v for v in leaks.values()), str(leaks))
        tr = obs["trades"]
        check("E observable trades cover every matched leg", len(tr) == len(sim.markets[0].matched_orders))
        check("E legs stay in clear order after the within-clear shuffle",
              bool(tr["time"].is_monotonic_increasing) and tr["leg_id"].tolist() == list(range(len(tr))))
        w = lab["wallets"]
        raw_codes = all(r.wallet == f"w{r.agent_id:03d}" for r in w.itertuples())
        check("E wallets are anonymized (codes are not raw agent ids)", not raw_codes)
        signed = np.where(tr["side"] == "SELL", -tr["qty"], tr["qty"])
        pos_obs = pd.Series(signed, index=tr["wallet"]).groupby(level=0).sum()
        pos_lab = w.set_index("wallet")["final_position"]
        check("E observable tape + wallet labels reproduce every final position",
              bool((pos_obs.reindex(pos_lab.index).fillna(0) == pos_lab).all()))
        legs = tr.merge(lab["trade_legs"], on=["leg_id", "wallet"])
        check("E leg labels join 1:1 and agree with order ids",
              len(legs) == len(tr) and bool((legs["is_wash_leg"] == legs["order_id"].map(is_wash_oid)).all()))
        check("E meta.json carries schema + config", meta["schema_version"].startswith("washtrade.v3")
              and "wash_groups" in meta["config"])

    # ------------------------------------------------------------------ F
    print("\n===== F) dataset generation =====")
    with tempfile.TemporaryDirectory() as tmp:
        man = generate("mini", seeds=2, backgrounds=("thin",), scenarios=("control", "triad"),
                       out_dir=tmp, verbose=False)
        root = os.path.join(tmp, "mini")
        check("F manifest has one row per run", len(man) == 4, str(man["run_id"].tolist()))
        check("F every run folder has observable/ labels/ meta.json",
              all(os.path.exists(os.path.join(root, p, "observable", "trades.csv"))
                  and os.path.exists(os.path.join(root, p, "labels", "wallets.csv"))
                  and os.path.exists(os.path.join(root, p, "meta.json")) for p in man["path"]))
        check("F control runs have zero wash events",
              bool((man.loc[man.scenario == "control", "n_events"] == 0).all()))

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
