r"""run_smoke.py -- v1 wash-trading smoke test (headless).

Run (from the repo root):
  python "capstone codes/washtrade/v1/run_smoke.py"

It checks the mechanism end-to-end, NOT the science:
  (A) all-ZI negative control      -> runs; zero wash events; market still trades.
  (B) ZI + 3 WashTrader treatment  -> wash events fire; the wash group's net position
      returns to ~0 at cycle boundaries and at end; wash self-trades show up in the trade
      log; per-agent net position is cleanly reconstructable from matched_orders.
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

from collections import defaultdict

import numpy as np

from washtrade.v1 import WashTradingSimulator
from marketsim.fourheap.constants import SELL


def seed_everything(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(composition, *, seed, lam_wash=None, wash_eps=0.0, sim_time=2000):
    seed_everything(seed)
    sim = WashTradingSimulator(
        composition, sim_time=sim_time, lam=5e-3,
        lam_wash=lam_wash, wash_eps=wash_eps, wash_seed=seed,
    )
    sim.run()
    return sim


def reconstruct_positions(sim):
    """Rebuild each agent's net position purely from the trade log (the Part-1 data pipeline)."""
    pos = defaultdict(int)
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        signed = (-o.quantity) if o.order_type == SELL else o.quantity
        pos[o.agent_id] += signed
    return pos


def wash_group_net(sim):
    return sum(sim.agents[i].position for i in sim.wash_wallet_ids)


def summarize(tag, sim):
    n_trades = len(sim.markets[0].matched_orders)
    log = sim.controller.event_log
    n_events = len(log)
    n_self = sum(1 for e in log if e["self_trade"])
    group_net = wash_group_net(sim)
    per_wallet = {i: sim.agents[i].position for i in sim.wash_wallet_ids}

    print(f"\n===== {tag} =====")
    print(f"  agents total            : {sim.num_agents}  "
          f"(wash wallets: {sim.wash_wallet_ids})")
    print(f"  matched orders (trades) : {n_trades}")
    print(f"  wash events fired       : {n_events}")
    if n_events:
        print(f"  self-trade success      : {n_self}/{n_events} "
              f"= {100.0 * n_self / n_events:.1f}%")
    print(f"  wash group net position : {group_net}  (target ~0)")
    print(f"  per-wallet final pos    : {per_wallet}")

    # cycle-boundary net-zero: after every full cycle of n wallets, cumulative signed volume
    # over the wash wallets should be 0 if all self-trades succeeded.
    if n_events:
        n_wallets = len(sim.wash_wallet_ids)
        cum = defaultdict(int)
        boundary_nets = []
        for k, e in enumerate(log, start=1):
            # realized leg fills (may be partial under a ZI interception)
            cum[e["seller"]] -= e["seller_filled"]
            cum[e["buyer"]] += e["buyer_filled"]
            if k % n_wallets == 0:
                boundary_nets.append(sum(cum[i] for i in sim.wash_wallet_ids))
        if boundary_nets:
            print(f"  cycle-boundary group net: "
                  f"min={min(boundary_nets)}, max={max(boundary_nets)} "
                  f"(over {len(boundary_nets)} completed cycles; target all 0)")
    return dict(n_trades=n_trades, n_events=n_events, n_self=n_self,
                group_net=group_net, per_wallet=per_wallet)


def main():
    SEED = 0
    # Use a brisker wash rate than the ZI reentry rate so we get many cycles in one smoke run.
    LAM_WASH = 0.05     # mean gap ~20 steps -> ~100 events -> ~33 full triangles over T=2000

    ctrl = run_one({"ZI": 14}, seed=SEED)                         # negative control
    trt = run_one({"ZI": 13, "WashTrader": 3}, seed=SEED, lam_wash=LAM_WASH)

    s_ctrl = summarize("A) all-ZI negative control", ctrl)
    s_trt = summarize("B) ZI + 3 WashTrader treatment", trt)

    # --- mechanism assertions (fail loudly if the plumbing is wrong) ---
    print("\n===== checks =====")
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    check("control fires zero wash events", s_ctrl["n_events"] == 0)
    check("control still trades (ZI liquidity)", s_ctrl["n_trades"] > 0)
    check("treatment fires wash events", s_trt["n_events"] > 0)
    check("treatment self-trades happen", s_trt["n_self"] > 0)
    check("wash group ends ~flat (|net| <= 1)", abs(s_trt["group_net"]) <= 1)

    # trade-log reconstruction must match the live positions the simulator booked
    recon = reconstruct_positions(trt)
    recon_ok = all(recon[i] == trt.agents[i].position for i in trt.wash_wallet_ids)
    check("positions reconstruct from matched_orders", recon_ok)

    print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
