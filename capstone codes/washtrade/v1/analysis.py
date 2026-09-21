r"""analysis.py -- post-run reconstruction & metrics for the v1 wash-trading mechanism.

Everything here is derived AFTER a run, from two sources:
  * ``sim.markets[0].matched_orders`` -- the trade log (each entry carries agent_id / side /
    qty / price / clear time). This is the Part-1 data pipeline: per-agent net position is
    rebuilt purely from here, no changes to marketsim/ needed.
  * ``sim.controller.event_log``     -- one row per fired wash event (intended pair + realized
    fills + self_trade flag). This is the ground-truth label stream (we KNOW who the wash
    wallets are and when they acted), which Part-2 detection will be scored against.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL


# --------------------------------------------------------------------------- #
# trade log -> tidy frame
# --------------------------------------------------------------------------- #
def trade_log_df(sim) -> pd.DataFrame:
    """One row per matched-order entry (a fill leg): time, agent_id, side, qty, price."""
    rows = []
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        side = "SELL" if o.order_type == SELL else "BUY"
        signed = (-o.quantity) if o.order_type == SELL else o.quantity
        rows.append(dict(time=mo.time, agent_id=o.agent_id, side=side,
                         qty=o.quantity, price=mo.price, signed_qty=signed))
    return pd.DataFrame(rows, columns=["time", "agent_id", "side", "qty", "price", "signed_qty"])


def reconstruct_positions(sim) -> dict:
    """Final net position per agent, rebuilt from the trade log alone."""
    pos = defaultdict(int)
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        pos[o.agent_id] += (-o.quantity) if o.order_type == SELL else o.quantity
    return dict(pos)


def position_timeseries(sim, agent_ids=None) -> pd.DataFrame:
    """Cumulative net position over time for the given agents (default: wash wallets).
    Index = 0..sim_time (forward-filled); columns = agent_ids."""
    if agent_ids is None:
        agent_ids = list(getattr(sim, "wash_wallet_ids", []))
    df = trade_log_df(sim)
    full_idx = pd.RangeIndex(0, sim.sim_time + 1, name="time")
    out = pd.DataFrame(index=full_idx)
    for aid in agent_ids:
        sub = df[df["agent_id"] == aid]
        series = sub.groupby("time")["signed_qty"].sum().cumsum()
        out[aid] = series.reindex(full_idx).ffill().fillna(0.0)
    return out


def wash_group_series(sim) -> pd.Series:
    """Net position of the whole wash group over time (should touch 0 at cycle boundaries)."""
    ts = position_timeseries(sim)
    if ts.shape[1] == 0:
        return pd.Series(0.0, index=pd.RangeIndex(0, sim.sim_time + 1, name="time"))
    return ts.sum(axis=1)


# --------------------------------------------------------------------------- #
# wash event log -> frame + summary
# --------------------------------------------------------------------------- #
def event_log_df(sim) -> pd.DataFrame:
    return pd.DataFrame(sim.controller.event_log)


def summarize(sim) -> dict:
    """Scalar mechanism metrics for one run."""
    log = sim.controller.event_log
    wash_ids = set(getattr(sim, "wash_wallet_ids", []))
    n_events = len(log)
    n_self = sum(1 for e in log if e["self_trade"])

    matched = sim.markets[0].matched_orders
    n_legs = len(matched)                                   # matched-order entries (2 per trade)
    wash_legs = sum(1 for mo in matched if mo.order.agent_id in wash_ids)

    return dict(
        n_wallets=len(wash_ids),
        n_events=n_events,
        n_self_trades=n_self,
        self_trade_rate=(n_self / n_events) if n_events else float("nan"),
        total_trade_legs=n_legs,
        wash_trade_legs=wash_legs,
        wash_leg_share=(wash_legs / n_legs) if n_legs else float("nan"),
        wash_group_final_net=int(sum(sim.agents[i].position for i in wash_ids)),
    )


def cycle_boundary_nets(sim) -> list:
    """Wash-group net (from realized fills) at each completed rotation boundary; target all 0."""
    log = sim.controller.event_log
    wash_ids = list(getattr(sim, "wash_wallet_ids", []))
    n = len(wash_ids)
    if n < 2 or not log:
        return []
    cum = defaultdict(int)
    nets = []
    for k, e in enumerate(log, start=1):
        cum[e["seller"]] -= e["seller_filled"]
        cum[e["buyer"]] += e["buyer_filled"]
        if k % n == 0:
            nets.append(int(sum(cum[i] for i in wash_ids)))
    return nets
