r"""analysis.py -- v6 run metrics: v3's, plus background activity, camouflage and leakage.

Everything in ``washtrade.v3.analysis`` works unchanged on a v6 simulator (same attributes) and
is re-exported here. v6 adds three groups of metrics:

* **background activity** -- is the honest market alive through the whole run, or only at the
  start and the end (the r = 0.01 pathology, see scenarios.py)?
* **camouflage** -- per group, how much of the wallets' volume is ordinary (non-wash) trading.
* **leakage** -- per group, the share of the group's traded volume whose counterparty is NOT a
  wallet of the same group. The clearing does not pair legs, so leakage is computed per clear
  under two pairing rules:

      pro-rata  (what the network detector sees):  within(c) = B_k(c) * S_k(c) / Q(c)
      max-match (most favourable to the group):     within(c) = min(B_k(c), S_k(c))

  where B_k / S_k are the group's bought / sold quantity in clear c and Q(c) the clear's total
  bought quantity. leak_k = 1 - sum_c 2 * within(c) / sum_c (B_k(c) + S_k(c)).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL

from washtrade.v3.analysis import (  # noqa: F401  (re-exported)
    cycle_boundary_nets, event_log_df, fill_log_df, lob_dense, lob_df, lob_levels_df,
    market_quality, order_log_df, position_timeseries, reconstruct_positions, trade_log_df,
)
from washtrade.v3.analysis import group_summary as _v3_group_summary
from washtrade.v3.analysis import summarize as _v3_summarize
from washtrade.v3.controller import is_wash_oid

N_TIME_BINS = 10


def _rate(num, den):
    return (num / den) if den else float("nan")


# --------------------------------------------------------------------------- #
# background activity
# --------------------------------------------------------------------------- #
def background_activity(sim, n_bins=N_TIME_BINS) -> dict:
    """Honest (non-wash-wallet) trading over time."""
    wallets = set(sim.wash_wallet_ids)
    background = [a for a in sim.agents if a not in wallets]
    times = np.array([mo.time for mo in sim.markets[0].matched_orders
                      if mo.order.agent_id not in wallets], dtype=float)
    counts = np.histogram(times, bins=n_bins, range=(0, sim.sim_time + 1))[0]
    lo, hi = sim.sim_time / 4, 3 * sim.sim_time / 4
    return dict(
        background_legs=int(len(times)),
        background_legs_per_agent=_rate(len(times), len(background)),
        background_active_bins=int((counts > 0).sum()),
        background_middle_share=_rate(int(((times >= lo) & (times < hi)).sum()), len(times)),
        background_legs_by_bin=counts.tolist(),
    )


# --------------------------------------------------------------------------- #
# per-group camouflage + leakage
# --------------------------------------------------------------------------- #
def _clear_flows(sim):
    """{time: (total bought qty, {group: [bought, sold]})} from the tape."""
    flows = defaultdict(lambda: [0.0, defaultdict(lambda: [0.0, 0.0])])
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        q = float(o.quantity)
        f = flows[mo.time]
        if o.order_type != SELL:
            f[0] += q
        k = sim.wallet_group.get(o.agent_id)
        if k is not None:
            f[1][k][1 if o.order_type == SELL else 0] += q
    return flows


def group_leakage(sim) -> dict:
    """{group: dict(volume, leak_prorata, leak_maxmatch)}; volume = bought + sold by the group."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0])       # volume, within pro-rata, within max-match
    for total_buy, by_group in _clear_flows(sim).values():
        for k, (b, s) in by_group.items():
            a = acc[k]
            a[0] += b + s
            if total_buy > 0:
                a[1] += b * s / total_buy
            a[2] += min(b, s)
    out = {}
    for k in range(len(sim.controllers)):
        vol, wp, wm = acc[k]
        out[k] = dict(volume=vol,
                      leak_prorata=(1.0 - 2.0 * wp / vol) if vol else float("nan"),
                      leak_maxmatch=(1.0 - 2.0 * wm / vol) if vol else float("nan"))
    return out


def group_summary(sim) -> pd.DataFrame:
    """v3's per-group table plus camouflage settings, ordinary-trade share and leakage."""
    df = _v3_group_summary(sim)
    if df.empty:
        return df
    camo = set(sim.camouflage_ids)
    ordinary = defaultdict(float)
    total = defaultdict(float)
    for mo in sim.markets[0].matched_orders:
        k = sim.wallet_group.get(mo.order.agent_id)
        if k is None:
            continue
        q = float(mo.order.quantity)
        total[k] += q
        if not is_wash_oid(mo.order.order_id):
            ordinary[k] += q
    leak = group_leakage(sim)
    df["n_camouflage"] = [sum(1 for i in c.wallet_ids if i in camo) for c in sim.controllers]
    df["camouflage_lam"] = [g.camouflage_lam for g in sim.groups]
    df["price_rule"] = [g.price_rule for g in sim.groups]
    df["ordinary_share"] = [_rate(ordinary[k], total[k]) for k in range(len(sim.controllers))]
    df["leak_prorata"] = [leak[k]["leak_prorata"] for k in range(len(sim.controllers))]
    df["leak_maxmatch"] = [leak[k]["leak_maxmatch"] for k in range(len(sim.controllers))]
    return df


def summarize(sim) -> dict:
    """v3's scalar metrics + background activity + pooled camouflage share and leakage."""
    s = _v3_summarize(sim)
    act = background_activity(sim)
    act.pop("background_legs_by_bin")
    s.update(act)
    s["r"] = float(sim.run_config["r"])
    s["lam"] = float(sim.lam)
    leak = group_leakage(sim)
    vol = sum(v["volume"] for v in leak.values())
    wallet_legs = [mo.order for mo in sim.markets[0].matched_orders
                   if mo.order.agent_id in sim.wallet_group]
    s["wallet_ordinary_share"] = _rate(
        sum(float(o.quantity) for o in wallet_legs if not is_wash_oid(o.order_id)),
        sum(float(o.quantity) for o in wallet_legs))
    s["leak_prorata"] = (sum(v["leak_prorata"] * v["volume"] for v in leak.values()) / vol
                         if vol else float("nan"))
    s["leak_maxmatch"] = (sum(v["leak_maxmatch"] * v["volume"] for v in leak.values()) / vol
                          if vol else float("nan"))
    return s
