r"""analysis.py -- post-run reconstruction and metrics for the v3 mechanism (in-memory sim).

Same sources as v2 -- ``matched_orders`` (the tape), each controller's ``event_log`` /
``fill_log`` / ``oid_meta`` (ground truth) and ``sim.lob`` -- but everything is multi-group aware
and wash volume is attributed by ORDER id, not by wallet: a camouflage wallet's ordinary ZI trades
are part of the tape but are not wash legs.

For the on-disk dataset (what Part 2 consumes) see ``export.py``.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL

from .controller import is_wash_oid


# --------------------------------------------------------------------------- #
# tape
# --------------------------------------------------------------------------- #
def trade_log_df(sim) -> pd.DataFrame:
    """One row per matched-order leg, with wallet- and order-level wash attribution."""
    wallets = set(sim.wash_wallet_ids)
    rows = []
    for i, mo in enumerate(sim.markets[0].matched_orders):
        o = mo.order
        meta = sim.wash_meta(o.order_id)
        rows.append(dict(
            idx=i, time=mo.time, agent_id=o.agent_id,
            side="SELL" if o.order_type == SELL else "BUY",
            qty=o.quantity, price=mo.price,
            signed_qty=(-o.quantity) if o.order_type == SELL else o.quantity,
            order_id=o.order_id,
            is_wash_wallet=o.agent_id in wallets,
            is_wash_leg=is_wash_oid(o.order_id),
            group_id=(meta["group_id"] if meta else sim.wallet_group.get(o.agent_id, -1)),
            event_idx=(meta["event_idx"] if meta else -1),
        ))
    return pd.DataFrame(rows, columns=["idx", "time", "agent_id", "side", "qty", "price",
                                       "signed_qty", "order_id", "is_wash_wallet",
                                       "is_wash_leg", "group_id", "event_idx"])


def reconstruct_positions(sim) -> dict:
    pos = defaultdict(int)
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        pos[o.agent_id] += (-o.quantity) if o.order_type == SELL else o.quantity
    return dict(pos)


def position_timeseries(sim, agent_ids=None) -> pd.DataFrame:
    """Cumulative net position over 0..sim_time (forward-filled); default: all wash wallets."""
    if agent_ids is None:
        agent_ids = list(sim.wash_wallet_ids)
    df = trade_log_df(sim)
    full_idx = pd.RangeIndex(0, sim.sim_time + 1, name="time")
    out = pd.DataFrame(index=full_idx)
    for aid in agent_ids:
        sub = df[df["agent_id"] == aid]
        out[aid] = sub.groupby("time")["signed_qty"].sum().cumsum().reindex(full_idx).ffill().fillna(0.0)
    return out


# --------------------------------------------------------------------------- #
# wash logs
# --------------------------------------------------------------------------- #
def event_log_df(sim) -> pd.DataFrame:
    frames = [pd.DataFrame(c.event_log) for c in sim.controllers if c.event_log]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fill_log_df(sim) -> pd.DataFrame:
    cols = ["time", "group_id", "order_id", "event_idx", "leg", "role", "agent_id",
            "side", "qty", "price", "post_time", "lag", "late"]
    rows = [r for c in sim.controllers for r in c.fill_log]
    return pd.DataFrame(rows, columns=cols)


def order_log_df(sim) -> pd.DataFrame:
    """Every wash order ever posted and how much of it filled (unfilled > 0 = withdrawn later)."""
    rows = []
    for c in sim.controllers:
        for oid, m in c.oid_meta.items():
            filled = float(c.filled_qty.get(oid, 0.0))
            rows.append(dict(order_id=oid, **m, filled=filled, unfilled=float(m["q"]) - filled))
    df = pd.DataFrame(rows)
    return df.sort_values("order_id").reset_index(drop=True) if len(df) else df


# --------------------------------------------------------------------------- #
# LOB
# --------------------------------------------------------------------------- #
def lob_df(sim) -> pd.DataFrame:
    return sim.lob.snapshots_df()


def lob_levels_df(sim) -> pd.DataFrame:
    return sim.lob.levels_df()


def lob_dense(sim, cols=("best_bid", "best_ask", "mid", "spread",
                         "bid_depth", "ask_depth")) -> pd.DataFrame:
    """Snapshots forward-filled onto 0..sim_time (exact: the book only changes on clears)."""
    df = lob_df(sim)
    full_idx = pd.RangeIndex(0, sim.sim_time + 1, name="time")
    if df.empty:
        return pd.DataFrame(index=full_idx, columns=list(cols), dtype=float)
    return df.set_index("time")[list(cols)].reindex(full_idx).ffill()


def market_quality(sim) -> dict:
    df = lob_df(sim)
    if df.empty:
        return {}
    two = df.dropna(subset=["mid"])
    nan = float("nan")
    return dict(
        n_clears=int(len(df)),
        mean_spread=float(two["spread"].mean()) if len(two) else nan,
        median_spread=float(two["spread"].median()) if len(two) else nan,
        mean_bid_depth=float(df["bid_depth"].mean()),
        mean_ask_depth=float(df["ask_depth"].mean()),
        n_steps_with_trade=int((df["n_matched_legs"] > 0).sum()),
    )


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _rate(num, den):
    return (num / den) if den else float("nan")


def group_summary(sim) -> pd.DataFrame:
    """One row per wash group."""
    rows = []
    for c, g in zip(sim.controllers, sim.groups):
        log = c.event_log
        posted = sum(m["q"] for m in c.oid_meta.values())
        filled = sum(c.filled_qty.get(oid, 0.0) for oid in c.oid_meta)
        rows.append(dict(
            group_id=c.group_id, n_wallets=len(c.wallet_ids), wallet_ids=list(c.wallet_ids),
            gap=c.gap, lam_wash=c.lam_wash, camouflage=g.camouflage,
            n_events=len(log),
            self_trade_rate=_rate(sum(e["self_trade"] for e in log), len(log)),
            intercept_rate=_rate(sum(e["intercepted"] for e in log), len(log)),
            leg_fill_rate=_rate(filled, posted),
            wash_volume=float(filled) / 2.0,
            final_net=int(sum(sim.agents[i].position for i in c.wallet_ids)),
            n_deferred=c.n_deferred,
        ))
    return pd.DataFrame(rows)


def summarize(sim) -> dict:
    """Scalar metrics for one run (all groups pooled)."""
    matched = sim.markets[0].matched_orders
    wallets = set(sim.wash_wallet_ids)
    camo = set(sim.camouflage_ids)
    log = [e for c in sim.controllers for e in c.event_log]

    total_qty = sum(float(mo.order.quantity) for mo in matched) / 2.0
    wash_leg_qty = sum(float(mo.order.quantity) for mo in matched if is_wash_oid(mo.order.order_id))
    wallet_qty = sum(float(mo.order.quantity) for mo in matched if mo.order.agent_id in wallets)
    posted = sum(m["q"] for c in sim.controllers for m in c.oid_meta.values())
    filled = sum(c.filled_qty.get(oid, 0.0) for c in sim.controllers for oid in c.oid_meta)

    return dict(
        n_groups=len(sim.controllers),
        n_wallets=len(wallets),
        n_camouflage_wallets=len(camo),
        n_background=len(sim.agents) - len(wallets),
        n_events=len(log),
        self_trade_rate=_rate(sum(e["self_trade"] for e in log), len(log)),
        intercept_rate=_rate(sum(e["intercepted"] for e in log), len(log)),
        leg_fill_rate=_rate(filled, posted),
        total_trade_legs=len(matched),
        wash_legs=sum(1 for mo in matched if is_wash_oid(mo.order.order_id)),
        camouflage_legit_legs=sum(1 for mo in matched if mo.order.agent_id in camo
                                  and not is_wash_oid(mo.order.order_id)),
        total_volume=total_qty,
        # a clear matches wash legs against each other or against outsiders, so wash volume is
        # counted per leg (/2 would assume both sides are wash)
        wash_leg_volume_share=_rate(wash_leg_qty, 2.0 * total_qty),
        wash_wallet_volume_share=_rate(wallet_qty, 2.0 * total_qty),
        n_deferred_opens=sum(c.n_deferred for c in sim.controllers),
        wash_group_final_net=int(sum(sim.agents[i].position for i in wallets)),
    )


def cycle_boundary_nets(sim, group_id=0) -> list:
    """Group net (realized as of close) at each completed rotation of ``group_id``; target 0."""
    c = sim.controllers[group_id]
    n = len(c.wallet_ids)
    cum = defaultdict(float)
    nets = []
    for k, e in enumerate(c.event_log, start=1):
        cum[e["seller"]] -= e["seller_filled"]
        cum[e["buyer"]] += e["buyer_filled"]
        if k % n == 0:
            nets.append(int(sum(cum[i] for i in c.wallet_ids)))
    return nets
