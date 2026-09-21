r"""analysis.py -- post-run reconstruction, metrics and disk export for the v2 mechanism.

Everything here is derived AFTER a run, from four sources:
  * ``sim.markets[0].matched_orders`` -- the trade tape (agent_id / side / qty / price / time).
    Per-agent net position is rebuilt purely from here, no changes to marketsim/ needed.
  * ``sim.controller.event_log``      -- one row per wash event (intent + realized outcome).
    This is the ground-truth label stream Part-2 detection will be scored against.
  * ``sim.controller.fill_log``       -- one row per realized fill of a wash order (order_id
    level), including fills that land after the event closed.
  * ``sim.lob``                       -- per-clear order-book snapshots (+ top-K depth).

``save_run`` dumps all four to CSV so a run can be analysed / shared without re-simulating.
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from marketsim.fourheap.constants import SELL


# --------------------------------------------------------------------------- #
# trade log -> tidy frame
# --------------------------------------------------------------------------- #
def trade_log_df(sim) -> pd.DataFrame:
    """One row per matched-order entry (a fill leg): time, agent_id, side, qty, price."""
    wash_ids = set(getattr(sim, "wash_wallet_ids", []))
    rows = []
    for mo in sim.markets[0].matched_orders:
        o = mo.order
        side = "SELL" if o.order_type == SELL else "BUY"
        signed = (-o.quantity) if o.order_type == SELL else o.quantity
        rows.append(dict(time=mo.time, agent_id=o.agent_id, side=side,
                         qty=o.quantity, price=mo.price, signed_qty=signed,
                         order_id=o.order_id, is_wash=o.agent_id in wash_ids))
    return pd.DataFrame(rows, columns=["time", "agent_id", "side", "qty", "price",
                                       "signed_qty", "order_id", "is_wash"])


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
# wash logs -> frames
# --------------------------------------------------------------------------- #
def event_log_df(sim) -> pd.DataFrame:
    return pd.DataFrame(sim.controller.event_log)


def fill_log_df(sim) -> pd.DataFrame:
    return pd.DataFrame(sim.controller.fill_log,
                        columns=["time", "order_id", "event_idx", "leg", "role", "agent_id",
                                 "side", "qty", "price", "post_time", "lag", "late"])


def order_log_df(sim) -> pd.DataFrame:
    """Every wash order ever posted, with how much of it eventually filled. Rows with
    ``filled < q`` are legs that rested unfilled and were later withdrawn."""
    c = sim.controller
    rows = []
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
    """LOB snapshots forward-filled onto a dense 0..sim_time index.

    Snapshots only exist for timesteps that actually cleared (upstream only calls
    ``market.step()`` on arrival/wash timesteps); in between, the book is unchanged, so a
    forward fill is exact rather than an approximation."""
    df = lob_df(sim)
    full_idx = pd.RangeIndex(0, sim.sim_time + 1, name="time")
    if df.empty:
        return pd.DataFrame(index=full_idx, columns=list(cols), dtype=float)
    return (df.set_index("time")[list(cols)]
              .reindex(full_idx).ffill())


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def summarize(sim) -> dict:
    """Scalar mechanism metrics for one run."""
    log = sim.controller.event_log
    wash_ids = set(getattr(sim, "wash_wallet_ids", []))
    q = sim.controller.q

    closed = [e for e in log if e.get("closed", True)]
    n_events = len(log)
    n_closed = len(closed)
    n_self = sum(1 for e in log if e["self_trade"])
    n_int = sum(1 for e in log if e.get("intercepted"))

    matched = sim.markets[0].matched_orders
    n_legs = len(matched)                                   # matched-order entries (2 per trade)
    wash_legs = sum(1 for mo in matched if mo.order.agent_id in wash_ids)
    total_qty = sum(float(mo.order.quantity) for mo in matched) / 2.0
    wash_qty = sum(float(mo.order.quantity) for mo in matched
                   if mo.order.agent_id in wash_ids) / 2.0

    posted = sum(m["q"] for m in sim.controller.oid_meta.values())
    filled = sum(sim.controller.filled_qty.get(oid, 0.0) for oid in sim.controller.oid_meta)

    return dict(
        n_wallets=len(wash_ids),
        wash_gap=sim.controller.gap,
        first_leg=sim.controller.first_leg,
        n_events=n_events,
        n_events_closed=n_closed,
        n_self_trades=n_self,
        self_trade_rate=(n_self / n_events) if n_events else float("nan"),
        n_intercepted=n_int,
        intercept_rate=(n_int / n_events) if n_events else float("nan"),
        wash_qty_posted=float(posted),
        wash_qty_filled=float(filled),
        leg_fill_rate=(filled / posted) if posted else float("nan"),
        total_trade_legs=n_legs,
        wash_trade_legs=wash_legs,
        wash_leg_share=(wash_legs / n_legs) if n_legs else float("nan"),
        total_volume=total_qty,
        wash_volume=wash_qty,
        wash_volume_share=(wash_qty / total_qty) if total_qty else float("nan"),
        wash_group_final_net=int(sum(sim.agents[i].position for i in wash_ids)),
    )


def market_quality(sim) -> dict:
    """Book-level summary (needs record_lob=True). Spread/depth are averaged over the
    post-clear snapshots; ``mid_vs_fundamental`` is the mean signed pricing error."""
    df = lob_df(sim)
    if df.empty:
        return {}
    two_sided = df.dropna(subset=["mid"])
    return dict(
        n_clears=int(len(df)),
        mean_spread=float(two_sided["spread"].mean()) if len(two_sided) else float("nan"),
        median_spread=float(two_sided["spread"].median()) if len(two_sided) else float("nan"),
        mean_bid_depth=float(df["bid_depth"].mean()),
        mean_ask_depth=float(df["ask_depth"].mean()),
        n_steps_with_trade=int((df["n_matched_legs"] > 0).sum()),
        mean_mid_vs_fundamental=(float((two_sided["mid"] - two_sided["fundamental"]).mean())
                                 if len(two_sided) else float("nan")),
    )


def cycle_boundary_nets(sim) -> list:
    """Wash-group net (from realized fills) at each completed rotation boundary; target all 0.

    With ``gap = 0`` every event self-trades, so all boundaries are exactly 0. With ``gap > 0``
    a boundary can be off by +-q when a leg was intercepted or left unfilled -- that deviation
    IS the inventory risk the gap introduces, so it is reported, not corrected.
    Fills that land after an event was closed are not counted here (see ``fill_log_df``)."""
    log = sim.controller.event_log
    wash_ids = list(getattr(sim, "wash_wallet_ids", []))
    n = len(wash_ids)
    if n < 2 or not log:
        return []
    cum = defaultdict(float)
    nets = []
    for k, e in enumerate(log, start=1):
        cum[e["seller"]] -= e["seller_filled"]
        cum[e["buyer"]] += e["buyer_filled"]
        if k % n == 0:
            nets.append(int(sum(cum[i] for i in wash_ids)))
    return nets


def event_timing_df(sim) -> pd.DataFrame:
    """Per-event timing view: when each leg was posted, when it actually filled, and the
    realized exposure of leg 1 (how long it sat in the book)."""
    ev = event_log_df(sim)
    if ev.empty:
        return ev
    fills = fill_log_df(sim)
    first_fill = (fills.groupby(["event_idx", "leg"])["time"].min().unstack()
                  if len(fills) else pd.DataFrame())
    out = ev[["event_idx", "open_time", "close_time", "gap", "first_leg", "seller", "buyer",
              "self_trade", "intercepted", "unfilled", "closed"]].copy()
    out["leg1_first_fill"] = (first_fill[1].reindex(out["event_idx"]).values
                              if 1 in getattr(first_fill, "columns", []) else np.nan)
    out["leg2_first_fill"] = (first_fill[2].reindex(out["event_idx"]).values
                              if 2 in getattr(first_fill, "columns", []) else np.nan)
    out["leg1_exposure"] = out["leg1_first_fill"] - out["open_time"]
    return out


# --------------------------------------------------------------------------- #
# disk export
# --------------------------------------------------------------------------- #
def save_run(sim, outdir, tag="run", extra_meta=None, save_levels=True) -> dict:
    """Write LOB snapshots, LOB depth levels, the trade tape, wash events, wash fills and a
    meta.json to ``outdir``, prefixed with ``tag``. Returns {name: path}.

    CSV (not parquet) on purpose: no pyarrow in this venv, and these files stay small enough
    (a T=2000 run is on the order of 10^3-10^4 rows per table)."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    def _dump(name, df):
        p = os.path.join(outdir, f"{tag}_{name}.csv")
        df.to_csv(p, index=False)
        paths[name] = p

    _dump("lob", lob_df(sim))
    if save_levels:
        _dump("lob_levels", lob_levels_df(sim))
    _dump("trades", trade_log_df(sim))
    _dump("wash_events", event_log_df(sim))
    _dump("wash_fills", fill_log_df(sim))
    _dump("wash_orders", order_log_df(sim))

    meta = dict(tag=tag,
                config=getattr(sim, "run_config", {}),
                summary=summarize(sim),
                market_quality=market_quality(sim),
                cycle_boundary_nets=cycle_boundary_nets(sim),
                final_positions={int(k): int(v) for k, v in reconstruct_positions(sim).items()})
    if extra_meta:
        meta.update(extra_meta)
    p = os.path.join(outdir, f"{tag}_meta.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)
    paths["meta"] = p
    return paths
