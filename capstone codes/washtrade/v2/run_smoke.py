r"""run_smoke.py -- v2 wash-trading smoke test (headless).

Run (from the repo root):
  python "capstone codes/washtrade/v2/run_smoke.py"

It checks the mechanism end-to-end, NOT the science:
  (A) all-ZI negative control        -> runs; zero wash events; market still trades.
  (B) ZI + 3 wallets, gap = 0        -> v1 behaviour: events fire, self-trade, net-zero cycling.
  (C) v1 equivalence                 -> v2 with wash_gap=0 reproduces washtrade.v1 exactly.
  (D) ZI + 2 wallets, gap = 0        -> the minimal A<->B structure works the same way.
  (E) ZI + wallets, gap > 0          -> legs really are separated in time; exposure/interception
                                        /unfilled outcomes are logged; group net stays bounded.
  (F) LOB recording + save_run       -> snapshots exist, the post-clear book is never crossed,
                                        and every table lands on disk.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAP = os.path.dirname(os.path.dirname(_HERE))     # .../capstone codes
_ROOT = os.path.dirname(_CAP)                       # .../<repo root>
for _p in (_ROOT, _CAP):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from washtrade.v2 import WashTradingSimulator
from washtrade.v2 import analysis as A


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(composition, *, seed, lam_wash=None, wash_eps=0.0, wash_gap=0,
            wash_first_leg="random", sim_time=2000, record_lob=True):
    seed_everything(seed)
    sim = WashTradingSimulator(
        composition, sim_time=sim_time, lam=5e-3,
        lam_wash=lam_wash, wash_eps=wash_eps, wash_gap=wash_gap,
        wash_first_leg=wash_first_leg, wash_seed=seed, record_lob=record_lob,
    )
    sim.run()
    return sim


def run_v1(composition, *, seed, lam_wash=None, wash_eps=0.0, sim_time=2000):
    from washtrade.v1 import WashTradingSimulator as V1Sim
    seed_everything(seed)
    sim = V1Sim(composition, sim_time=sim_time, lam=5e-3,
                lam_wash=lam_wash, wash_eps=wash_eps, wash_seed=seed)
    sim.run()
    return sim


def report(tag, sim):
    s = A.summarize(sim)
    nets = A.cycle_boundary_nets(sim)
    print(f"\n===== {tag} =====")
    print(f"  agents total            : {sim.num_agents}  (wash wallets: {sim.wash_wallet_ids})")
    print(f"  matched order legs      : {s['total_trade_legs']}")
    print(f"  wash events fired       : {s['n_events']} (closed: {s['n_events_closed']})")
    if s["n_events"]:
        print(f"  self-trade success      : {s['n_self_trades']}/{s['n_events']} "
              f"= {100 * s['self_trade_rate']:.1f}%")
        print(f"  intercepted events      : {s['n_intercepted']}/{s['n_events']} "
              f"= {100 * s['intercept_rate']:.1f}%")
        print(f"  wash leg fill rate      : {100 * s['leg_fill_rate']:.1f}% "
              f"({s['wash_qty_filled']:.0f}/{s['wash_qty_posted']:.0f} units)")
        print(f"  wash volume share       : {100 * s['wash_volume_share']:.1f}%")
    print(f"  wash group net position : {s['wash_group_final_net']}  (target ~0)")
    print(f"  per-wallet final pos    : {{{', '.join(f'{i}: {sim.agents[i].position}' for i in sim.wash_wallet_ids)}}}")
    if nets:
        print(f"  cycle-boundary group net: min={min(nets)}, max={max(nets)} "
              f"(over {len(nets)} cycles; target all 0)")
    mq = A.market_quality(sim)
    if mq:
        print(f"  LOB snapshots           : {mq['n_clears']} rows | "
              f"mean spread={mq['mean_spread']:.1f} | "
              f"mean depth (bid/ask)={mq['mean_bid_depth']:.1f}/{mq['mean_ask_depth']:.1f}")
    return s


def main():
    SEED = 0
    LAM_WASH = 0.05     # mean gap ~20 steps -> ~100 events over T=2000
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")

    # ---------------- A / B / D: gap = 0 conditions ---------------- #
    ctrl = run_one({"ZI": 14}, seed=SEED)
    trt3 = run_one({"ZI": 13, "WashTrader": 3}, seed=SEED, lam_wash=LAM_WASH, wash_gap=0)
    trt2 = run_one({"ZI": 13, "WashTrader": 2}, seed=SEED, lam_wash=LAM_WASH, wash_gap=0)

    s_ctrl = report("A) all-ZI negative control", ctrl)
    s_trt3 = report("B) ZI + 3 wallets, gap = 0", trt3)
    s_trt2 = report("D) ZI + 2 wallets, gap = 0", trt2)

    # ---------------- E: gapped conditions ---------------- #
    gapped = {g: run_one({"ZI": 13, "WashTrader": 3}, seed=SEED, lam_wash=LAM_WASH, wash_gap=g)
              for g in (1, 5, 20)}
    for g, sim in gapped.items():
        report(f"E) ZI + 3 wallets, gap = {g}", sim)
    g2 = run_one({"ZI": 13, "WashTrader": 2}, seed=SEED, lam_wash=LAM_WASH, wash_gap=5)
    report("E) ZI + 2 wallets, gap = 5", g2)

    print("\n===== checks =====")
    check("A control fires zero wash events", s_ctrl["n_events"] == 0)
    check("A control still trades (ZI liquidity)", s_ctrl["total_trade_legs"] > 0)
    check("A control records a LOB", A.lob_df(ctrl).shape[0] > 0,
          f"{A.lob_df(ctrl).shape[0]} snapshots")

    for tag, s, sim in (("B 3-wallet", s_trt3, trt3), ("D 2-wallet", s_trt2, trt2)):
        check(f"{tag} gap=0 fires wash events", s["n_events"] > 0, f"{s['n_events']} events")
        check(f"{tag} gap=0 self-trades ~always", s["self_trade_rate"] > 0.9,
              f"{s['self_trade_rate']:.3f}")
        check(f"{tag} gap=0 group ends ~flat", abs(s["wash_group_final_net"]) <= 1)
        recon = A.reconstruct_positions(sim)
        check(f"{tag} positions reconstruct from matched_orders",
              all(recon.get(i, 0) == sim.agents[i].position for i in sim.wash_wallet_ids))
        ev = A.event_log_df(sim)
        check(f"{tag} gap=0 legs are simultaneous",
              bool((ev["open_time"] == ev["close_time"]).all()))

    # 2-wallet specific: exactly one reciprocal edge pair, cycle length 2
    ev2 = A.event_log_df(trt2)
    edges2 = set(zip(ev2["seller"], ev2["buyer"]))
    a, b = trt2.wash_wallet_ids
    check("D 2-wallet uses exactly the A<->B reciprocal pair",
          edges2 == {(a, b), (b, a)}, str(sorted(edges2)))
    nets2 = A.cycle_boundary_nets(trt2)
    check("D 2-wallet nets to 0 every 2 events", bool(nets2) and max(abs(x) for x in nets2) == 0)

    # ---------------- C: v1 equivalence at gap = 0 ---------------- #
    v1 = run_v1({"ZI": 13, "WashTrader": 3}, seed=SEED, lam_wash=LAM_WASH)
    v1_events = len(v1.controller.event_log)
    v1_self = sum(1 for e in v1.controller.event_log if e["self_trade"])
    v1_pos = {i: v1.agents[i].position for i in v1.wash_wallet_ids}
    v2_pos = {i: trt3.agents[i].position for i in trt3.wash_wallet_ids}
    v1_trades = len(v1.markets[0].matched_orders)
    same = (v1_events == s_trt3["n_events"] and v1_self == s_trt3["n_self_trades"]
            and v1_pos == v2_pos and v1_trades == s_trt3["total_trade_legs"])
    check("C v2(gap=0) reproduces v1 exactly", same,
          f"v1: {v1_events} ev / {v1_self} self / {v1_trades} legs | "
          f"v2: {s_trt3['n_events']} / {s_trt3['n_self_trades']} / {s_trt3['total_trade_legs']}")
    # same event-by-event intent (edges and prices), not just the totals
    e1 = A.event_log_df(v1) if hasattr(v1, "controller") else None
    e2 = A.event_log_df(trt3)
    same_prices = (len(v1.controller.event_log) == len(e2) and all(
        abs(a_["anchor"] - b_) < 1e-9 and a_["seller"] == c_ and a_["buyer"] == d_
        for a_, b_, c_, d_ in zip(v1.controller.event_log, e2["anchor"], e2["seller"], e2["buyer"])))
    check("C v2(gap=0) matches v1 edge-by-edge and price-by-price", same_prices)

    # ---------------- E checks ---------------- #
    for g, sim in gapped.items():
        s = A.summarize(sim)
        ev = A.event_log_df(sim)
        closed = ev[ev["closed"]]
        check(f"E gap={g}: legs are exactly {g} steps apart",
              bool((closed["close_time"] - closed["open_time"] == g).all()))
        check(f"E gap={g}: events still fire", s["n_events"] > 0, f"{s['n_events']} events")
        check(f"E gap={g}: group net stays bounded (|net| <= 2q)",
              abs(s["wash_group_final_net"]) <= 2 * sim.controller.q,
              f"net={s['wash_group_final_net']}")
        check(f"E gap={g}: first leg is randomised over events",
              set(closed["first_leg"]) <= {"SELL", "BUY"} and len(set(closed["first_leg"])) == 2,
              str(sorted(set(closed["first_leg"]))))
        # exposure: a leg-1 fill can now happen strictly before the partner arrives
        tim = A.event_timing_df(sim)
        n_early = int((tim["leg1_first_fill"] < tim["close_time"]).sum())
        print(f"        gap={g}: interceptions={s['n_intercepted']}, "
              f"leg-1 filled before close={n_early}, "
              f"leg fill rate={100 * s['leg_fill_rate']:.1f}%, "
              f"self-trade rate={100 * s['self_trade_rate']:.1f}%")

    check("E gap monotonically hurts the self-trade rate",
          A.summarize(gapped[1])["self_trade_rate"] >= A.summarize(gapped[20])["self_trade_rate"],
          f"gap1={A.summarize(gapped[1])['self_trade_rate']:.3f} "
          f"gap20={A.summarize(gapped[20])['self_trade_rate']:.3f}")

    # ---------------- F: LOB integrity + export ---------------- #
    lob = A.lob_df(gapped[5])
    both = lob.dropna(subset=["best_bid", "best_ask"])
    check("F post-clear book is never crossed", bool((both["best_bid"] <= both["best_ask"]).all()))
    check("F LOB has one row per cleared timestep", lob["time"].is_unique and len(lob) > 0,
          f"{len(lob)} rows")
    lv = A.lob_levels_df(gapped[5])
    check("F depth levels recorded", len(lv) > 0, f"{len(lv)} level-rows")
    check("F wash legs are flagged in the LOB rows", int(lob["wash_legs"].sum()) > 0)

    outdir = os.path.join(_HERE, "data", "smoke")
    paths = A.save_run(gapped[5], outdir, tag="smoke_gap5")
    sizes = {k: os.path.getsize(v) for k, v in paths.items()}
    check("F save_run wrote every table", all(v > 0 for v in sizes.values()),
          ", ".join(f"{k}={v}B" for k, v in sizes.items()))
    print(f"        saved to: {outdir}")

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
