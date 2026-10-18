r"""analysis.py -- v8 run metrics: v7's, plus what the v8 knobs cost the adversary and what they
hide from a short-window detector.

Everything in ``washtrade.v7.analysis`` works unchanged on a v8 simulator and is re-exported.
v8 adds, per wash group:

* **execution** -- ``edge_order``, ``pipeline``, the largest number of rounds open at once, and the
  share of started rounds that completed before the run ended (``rounds_completed_share``).
* **what a detector sees** -- ``repeat_share_50``: the share of wash events whose ordered pair
  (seller -> buyer) has another wash event within 50 steps. This is the ground-truth version of
  detection v4's ``repeat_50`` channel. ``hold_mean`` is the mean time, in steps, that a wash
  wallet's unit position stays open before an opposite unit undoes it: all of the wallet's legs,
  matched first-in first-out, as detection v4's round-trip channel matches them. Units still
  open at the end of the run are not in the mean; their share is ``open_unit_share``.
* **what it costs** -- holding is the cost: a wallet that holds longer carries the price risk
  longer (``hold_mean``), and positions still open at the end are unhedged at wallet level
  (``open_unit_share``; the group as a whole stays flat). Wash volume per 1000 steps
  (``wash_volume_rate``) is what the scheme buys.

And per run: ``honest_mean_order_size`` (the sizes multi-unit honest agents drew; 1 for plain ZI)
and ``honest_mean_leg_qty`` (mean quantity of a filled honest leg on the tape).
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL

from washtrade.v3.controller import is_wash_oid
from washtrade.v7.analysis import (  # noqa: F401  (re-exported)
    CLEAR_TIME_OFFSET, background_activity, cycle_boundary_nets, event_log_df, fill_log_df,
    group_leakage, lob_dense, lob_df, lob_levels_df, market_quality, order_log_df,
    position_timeseries, reconstruct_positions, round_end_positions, round_log_df,
    trade_log_df, zero_returns,
)
from washtrade.v7.analysis import group_summary as _v7_group_summary
from washtrade.v7.analysis import summarize as _v7_summarize

REPEAT_WINDOW = 50


def _rate(num, den):
    return (num / den) if den else float("nan")


def _steps(sim):
    """agent_id -> [(time, net signed quantity)] per clear (nonzero only), in time order."""
    by = defaultdict(lambda: defaultdict(int))
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        by[o.agent_id][mo.time] += (-o.quantity) if o.order_type == SELL else o.quantity
    return {a: [(t, q) for t, q in sorted(d.items()) if q != 0] for a, d in by.items()}


def fifo_holds(steps):
    """(hold times of matched units, number of units left open) for one wallet's steps."""
    open_ = deque()                       # (time, sign) of unmatched units, all of one sign
    holds = []
    for t, q in steps:
        s = 1 if q > 0 else -1
        for _ in range(int(abs(q))):
            if open_ and open_[0][1] == -s:
                holds.append(t - open_.popleft()[0])
            else:
                open_.append((t, s))
    return holds, len(open_)


def repeat_share(events, window=REPEAT_WINDOW) -> float:
    """Share of events whose ordered (seller, buyer) pair has another event within ``window``."""
    by = defaultdict(list)
    for e in events:
        by[(e["seller"], e["buyer"])].append(e["open_time"])
    hit = total = 0
    for times in by.values():
        times.sort()
        for i, t in enumerate(times):
            total += 1
            near = (i > 0 and t - times[i - 1] <= window) or \
                   (i + 1 < len(times) and times[i + 1] - t <= window)
            hit += bool(near)
    return _rate(hit, total)


def group_summary(sim) -> pd.DataFrame:
    df = _v7_group_summary(sim)
    if df.empty:
        return df
    steps = _steps(sim)
    vol = defaultdict(float)
    for mo in sim.markets[0].matched_orders:
        if is_wash_oid(mo.order.order_id):
            vol[sim.wash_meta(mo.order.order_id)["group_id"]] += float(mo.order.quantity)
    rows = []
    for c in sim.controllers:
        used = [w for w in c.wallet_ids if c.rounds_used[w] > 0]
        holds, n_open, n_units = [], 0, 0
        for w in used:
            h, o = fifo_holds(steps.get(w, []))
            holds += h
            n_open += o
            n_units += 2 * len(h) + o
        done = defaultdict(int)
        for e in c.event_log:
            done[e["round_idx"]] += 1
        rl = c.round_log
        rows.append(dict(
            edge_order=getattr(c, "edge_order", "sequential"),
            pipeline=getattr(c, "pipeline", 1),
            max_open_rounds=getattr(c, "max_open_rounds", 1),
            rounds_completed_share=_rate(sum(done[r["round_idx"]] == r["n_events"] for r in rl),
                                         len(rl)),
            repeat_share_50=repeat_share(c.event_log),
            hold_mean=float(np.mean(holds)) if holds else float("nan"),
            hold_median=float(np.median(holds)) if holds else float("nan"),
            open_unit_share=_rate(n_open, n_units),
            wash_volume_rate=1000.0 * vol[c.group_id] / 2.0 / sim.sim_time,
        ))
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


def summarize(sim) -> dict:
    """v7's scalar metrics + honest order sizes + pooled v8 group metrics."""
    s = _v7_summarize(sim)
    sizes = sim.honest_order_sizes() if hasattr(sim, "honest_order_sizes") else []
    s["order_size_max"] = getattr(sim, "order_size_max", None)
    s["honest_mean_order_size"] = float(np.mean(sizes)) if sizes else 1.0
    wallets = set(sim.wash_wallet_ids)
    legs = [float(mo.order.quantity) for mo in sim.markets[0].matched_orders
            if mo.order.agent_id not in wallets]
    s["honest_mean_leg_qty"] = float(np.mean(legs)) if legs else float("nan")
    g = group_summary(sim)
    if len(g):
        for col in ("rounds_completed_share", "repeat_share_50", "hold_mean", "hold_median",
                    "open_unit_share", "wash_volume_rate", "max_open_rounds"):
            s[col] = float(g[col].mean())
    return s
